"""Les tests les plus importants du projet.

Une erreur dans le terme proximal ne produit ni exception ni courbe absurde :
elle produit des résultats plausibles et faux. Si `mu = 0` ne reproduit pas
FedAvg exactement, ou si le terme s'annule silencieusement, la comparaison
FedAvg/FedProx — le livrable central — est à jeter sans que rien ne l'ait
signalé.

**Ces tests comparent les POIDS, pas l'accuracy.** Le guide proposait
l'accuracy ; mesure faite, c'est un observable trop grossier. Sur le jeu
injecté, FedAvg et FedProx affichent 0,1 partout — le modèle est au niveau du
hasard — alors que leurs poids diffèrent de 5,3e-4. Un test sur l'accuracy
passerait donc sans rien vérifier. Sur le vrai MNIST le signal existe, mais il
reste quantifié par la taille du jeu de test : les poids sont le seul
observable exact.
"""

import torch
from torch.utils.data import DataLoader

from contracts.schemas import Algo, RunConfig
from fl_core.models.cnn import build_model
from fl_core.server import run_federated
from fl_core.train import train_local

from .conftest import jeu_minuscule


def _modele_final(algo: Algo, mu: float):
    """Le modèle global après 5 rounds. Les poids sont l'observable exact."""
    cfg = RunConfig(algo=algo, mu=mu, alpha=0.5, n_clients=4,
                    rounds=5, local_epochs=1, batch_size=32, seed=0)
    return run_federated("t", cfg, lambda m: None)


def _ecart_max(a, b) -> float:
    return max(float((p - q).abs().max().detach())
               for p, q in zip(a.parameters(), b.parameters()))


def test_mu_zero_reproduit_fedavg(sans_mnist):
    """À mu = 0, le bloc du terme proximal n'est pas exécuté : le chemin de
    code est STRICTEMENT identique à FedAvg.

    Ce n'est donc pas une approximation numérique mais une identité — même
    seed, mêmes tirages, mêmes opérations flottantes dans le même ordre. On
    exige l'égalité bit à bit, pas une tolérance.

    C'est ce qui autorise l'affirmation du rapport : entre les deux bras, une
    seule variable change.
    """
    a = _modele_final(Algo.fedavg, 0.0)
    b = _modele_final(Algo.fedprox, 0.0)

    for (nom, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), f"{nom} diffère alors que mu = 0"


def test_mu_positif_change_le_resultat(sans_mnist):
    """Attrape l'annulation silencieuse.

    Si `global_params` était mal transmis — None, ou rafraîchi de sorte que
    w - w^t vaille zéro — le terme proximal disparaîtrait et FedProx
    redeviendrait FedAvg. Aucune erreur, aucun crash, des courbes plausibles.

    Seuil : l'écart mesuré est de 5,3e-4, on exige 1e-6. Large marge, mais
    strictement au-dessus du bruit flottant.
    """
    a = _modele_final(Algo.fedavg, 0.0)
    b = _modele_final(Algo.fedprox, 1.0)

    assert _ecart_max(a, b) > 1e-6


def test_un_mu_fort_retient_le_modele_local_pres_du_global(sans_mnist):
    """Le SENS du terme, que les deux tests précédents ne vérifient pas.

    Avec `loss - (mu/2)*prox` au lieu de `loss + ...`, le ressort repousserait
    le modèle local LOIN du global. Le test précédent passerait quand même —
    le résultat change bien — mais FedProx ferait l'exact contraire de ce que
    décrit le papier, et l'accuracy s'effondrerait sans explication.

    Mesuré sur le jeu injecté, distance ||w - w^t||² après une époque :
    mu=0 -> 1,00e-2 · mu=10 -> 6,35e-3 · mu=100 -> 4,77e-4.

    ATTENTION au choix de mu : au-delà, `lr * mu` dépasse la limite de
    stabilité de SGD et l'itération DIVERGE (mu=1000 -> 1,5e+2,
    mu=10000 -> 5,0e+8). Un mu trop grand ne fige donc pas le modèle, il le
    fait exploser. C'est une contrainte à respecter dans le balayage de mu.
    """
    loader = DataLoader(jeu_minuscule().train, batch_size=32, shuffle=False)

    def distance_parcourue(mu: float) -> float:
        torch.manual_seed(0)
        model = build_model("mnist")
        ancre = [p.detach().clone() for p in model.parameters()]
        train_local(model, loader, epochs=1, lr=0.01, mu=mu, global_params=ancre)
        return float(sum(
            ((p - g) ** 2).sum().detach() for p, g in zip(model.parameters(), ancre)
        ))

    libre = distance_parcourue(0.0)
    retenu = distance_parcourue(100.0)

    assert libre > 0, "sans terme proximal, le modèle doit bouger"
    assert retenu < libre / 10, (
        f"le terme proximal ne retient pas : {retenu:.3e} contre {libre:.3e} "
        "en chute libre — vérifier le signe"
    )
