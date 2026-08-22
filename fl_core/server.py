"""La boucle fédérée : FedAvg, et FedProx sans une ligne de plus.

Les deux algorithmes partagent strictement ce fichier. Leur seule différence
est la valeur de `cfg.mu`, transmise telle quelle à `train_local` : à mu = 0 le
bloc du terme proximal n'est pas exécuté et le chemin de code est identique à
FedAvg. C'est ce qui rend la comparaison propre — une seule variable change
entre les deux bras — et c'est vérifiable, pas seulement affirmé.

Cette implémentation manuelle a deux raisons d'exister. D'abord servir
d'oracle : au portage sur Flower, des courbes qui se superposent valident les
deux côtés à la fois, là où un écart n'aurait autrement aucune cause
identifiable entre modèle, partition et configuration Flower. Ensuite servir
de substrat au mode asynchrone, que Flower en simulation n'exprime pas.
"""

from __future__ import annotations

import copy
import random
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from contracts.schemas import ClientMetric, RoundMetric, RunConfig
from fl_core.aggregate import average_weights
from fl_core.data.loaders import load
from fl_core.data.partition import dirichlet_partition
from fl_core.models.cnn import build_model
from fl_core.seeding import seed_everything
from fl_core.train import evaluate, model_size_mb, train_local


def resolve_device(prefere: str | None = None) -> str:
    """Le GPU s'il y en a un, sinon le CPU. Un choix explicite l'emporte.

    Le device ne vit PAS dans `RunConfig` : c'est une propriété de la machine
    qui exécute, pas de l'expérience. Deux exécutions d'une même config sur CPU
    et sur GPU doivent rester la même expérience, donc le même run.
    """
    if prefere:
        return prefere
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_federated(run_id: str, cfg: RunConfig, on_round, device: str | None = None):
    """Exécute `cfg.rounds` rounds de communication et émet un RoundMetric par round.

    Retourne le modèle global entraîné.
    """
    seed_everything(cfg.seed)
    device = resolve_device(device)

    split = load(cfg.dataset.value)
    parts, _ = dirichlet_partition(
        split.labels, cfg.n_clients, cfg.alpha, seed=cfg.seed
    )
    sizes = [len(idx) for idx in parts]

    # Construits une seule fois : les reconstruire à chaque round coûterait
    # cher sans rien changer au résultat.
    loaders = [
        DataLoader(Subset(split.train, idx), batch_size=cfg.batch_size, shuffle=True)
        for idx in parts
    ]
    test_loader = DataLoader(split.test, batch_size=512)

    # .to(device) AVANT la boucle : l'ancre proximale est clonée depuis ce
    # modèle, et les modèles locaux sont déplacés par `train_local`. Si le
    # global restait sur CPU, le terme proximal comparerait un tenseur CPU à un
    # tenseur GPU et lèverait à la première itération de FedProx.
    global_model = build_model(cfg.dataset.value).to(device)
    mb_par_client = model_size_mb(global_model)

    n_selected = max(1, round(cfg.n_clients * cfg.participation))
    rng = np.random.default_rng(cfg.seed)

    t0 = time.time()
    for r in range(1, cfg.rounds + 1):
        selection = rng.choice(cfg.n_clients, size=n_selected, replace=False)

        # ANCRE DU TERME PROXIMAL — capturée UNE SEULE FOIS, avant la boucle
        # clients. La rafraîchir par client (ou pire, passer les paramètres du
        # modèle local) annulerait le terme : FedProx redeviendrait FedAvg sans
        # qu'aucune erreur ne le signale, et les courbes resteraient plausibles.
        global_params = [p.detach().clone() for p in global_model.parameters()]

        states, poids, accs_locales = [], [], []
        clients: list[ClientMetric] = []

        for k in selection:
            t_client = time.time()
            local = copy.deepcopy(global_model)          # broadcast

            # Hétérogénéité systèmes : le straggler rend un travail partiel au
            # lieu d'être exclu du round. C'est le second apport du papier
            # FedProx, et le régime où le terme proximal compte le plus.
            epochs_k = (
                random.randint(1, cfg.local_epochs)
                if cfg.systems_heterogeneity
                else cfg.local_epochs
            )

            train_local(
                local, loaders[k], epochs_k, cfg.lr, device=device,
                mu=cfg.mu, global_params=global_params,
            )

            # Sur le jeu de test GLOBAL : seule mesure comparable entre clients,
            # et avec le modèle global. Un client évalué sur ses propres classes
            # afficherait un chiffre flatteur qui ne dit rien.
            acc_locale, loss_locale = evaluate(local, test_loader, device)

            # ||w_k - w^t||_2 : exactement la quantité que le terme proximal
            # pénalise. La mesurer par client donne la preuve directe que
            # FedProx CONTIENT la dérive, là où l'accuracy ne montre que son
            # effet supposé sur le score.
            drift = float(torch.sqrt(sum(
                ((p - g) ** 2).sum()
                for p, g in zip(local.parameters(), global_params)
            )).detach())

            accs_locales.append(acc_locale)
            states.append(local.state_dict())
            poids.append(sizes[k])
            clients.append(ClientMetric(
                client_id=int(k),
                n_samples=sizes[k],
                epochs_run=epochs_k,
                local_acc=acc_locale,
                local_loss=loss_locale,
                drift=drift,
                wall_time_s=round(time.time() - t_client, 2),
            ))

        global_model.load_state_dict(average_weights(states, poids))
        acc, loss = evaluate(global_model, test_loader, device)

        on_round(RoundMetric(
            run_id=run_id, round=r,
            global_acc=acc, global_loss=loss,
            mean_client_acc=statistics.fmean(accs_locales),
            # Dispersion des accuracies locales : la mesure directe du client
            # drift, obtenue gratuitement puisque chaque local est déjà évalué.
            std_client_acc=(
                statistics.stdev(accs_locales) if len(accs_locales) > 1 else 0.0
            ),
            # Cumulatif : descente + remontée, pour chaque client sélectionné.
            comm_mb=round(2 * n_selected * mb_par_client * r, 2),
            wall_time_s=round(time.time() - t0, 2),
            clients=clients,
        ))

    return global_model
