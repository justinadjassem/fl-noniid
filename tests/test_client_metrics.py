"""Monitoring par client : le grain fin du drift.

Jusqu'ici on ne mesurait que l'agrégat par round. `ClientMetric` descend au
client, et sa métrique centrale — `drift` — est **exactement la quantité que le
terme proximal de FedProx pénalise**. La journaliser donne bien plus qu'une
courbe d'accuracy : la preuve directe que FedProx contient la dérive, et pas
seulement qu'il améliore le score.
"""

import statistics

from contracts.schemas import Algo, ClientMetric, RoundMetric, RunConfig
from fl_core.server import run_federated

BASE = dict(alpha=0.5, n_clients=4, rounds=2, local_epochs=1,
            batch_size=32, seed=0)


def _rounds(**surcharges) -> list[RoundMetric]:
    m = []
    run_federated("t", RunConfig(**{**BASE, **surcharges}), m.append)
    return m


def test_le_champ_clients_est_retrocompatible():
    """Défaut vide : le moteur factice, les bornes et le dashboard continuent
    de fonctionner sans une ligne modifiée. C'est la forme à privilégier pour
    toute évolution du contrat."""
    m = RoundMetric(run_id="x", round=1, global_acc=0.5, global_loss=1.0,
                    mean_client_acc=0.5, std_client_acc=0.0)

    assert m.clients == []


def test_un_client_metric_par_client_selectionne(sans_mnist):
    m = _rounds(algo=Algo.fedavg)

    for r in m:
        assert len(r.clients) == 4
        assert sorted(c.client_id for c in r.clients) == [0, 1, 2, 3]
        assert all(isinstance(c, ClientMetric) for c in r.clients)


def test_la_participation_partielle_ne_journalise_que_les_selectionnes(sans_mnist):
    m = _rounds(algo=Algo.fedavg, participation=0.5)

    assert all(len(r.clients) == 2 for r in m)


def test_les_champs_decrivent_reellement_le_client(sans_mnist):
    """n_samples porte le déséquilibre de quantité — le second visage du
    non-IID — et doit donc refléter la vraie taille du shard, pas une moyenne."""
    m = _rounds(algo=Algo.fedavg, local_epochs=2)
    clients = m[0].clients

    assert all(c.n_samples > 0 for c in clients)
    assert sum(c.n_samples for c in clients) == 120, "les shards partitionnent le jeu"
    assert len({c.n_samples for c in clients}) > 1, "à alpha=0.5 les tailles diffèrent"
    assert all(c.epochs_run == 2 for c in clients)
    assert all(0.0 <= c.local_acc <= 1.0 for c in clients)
    assert all(c.drift > 0 for c in clients), "un client qui s'entraîne s'éloigne"
    assert all(c.wall_time_s >= 0 for c in clients)


def test_epochs_run_reflete_l_heterogeneite_systemes(sans_mnist):
    """`epochs_run` est ce qui révèle les stragglers dans le dashboard :
    sans lui, on ne saurait pas quel client a rendu un travail partiel."""
    m = _rounds(algo=Algo.fedavg, local_epochs=4, n_clients=6,
                systems_heterogeneity=True)
    epoques = [c.epochs_run for c in m[0].clients]

    assert len(set(epoques)) > 1, f"toutes identiques : {epoques}"
    assert all(1 <= e <= 4 for e in epoques)


def test_le_terme_proximal_reduit_le_drift(sans_mnist):
    """POINT D'ARRÊT ML-8.

    `drift` est la norme que FedProx pénalise. À alpha égal et seed égale, un
    mu positif doit produire un drift visiblement plus faible qu'un mu nul.

    C'est un test plus fin que la comparaison d'accuracy : il vérifie le
    MÉCANISME annoncé — contenir la dérive — et pas seulement son effet
    supposé sur le score.

    ATTENTION AU RÉGIME. Le terme proximal n'agit qu'APRÈS que le modèle local
    a bougé : au premier pas w = w^t, donc le gradient proximal mu*(w - w^t)
    vaut zéro. Son influence croît donc avec le nombre de pas locaux. Mesuré
    ici, réduction du drift à mu=10 :

        1 pas -> 2,3 %   ·   2 pas -> 12,3 %   ·   3 pas -> 45,5 %
        15 pas -> 67,7 %  ·  30 pas -> 73,0 %

    Autrement dit, **à une seule étape locale FedProx est identique à FedAvg
    par construction**. D'où le choix de 3 époques en batch 16 ci-dessous : il
    faut se placer dans un régime où l'effet existe. La grille finale, avec ses
    188 pas locaux par client et par round, y est largement.
    """
    def drift_moyen(mu: float) -> float:
        m = _rounds(algo=Algo.fedprox if mu else Algo.fedavg, mu=mu,
                    local_epochs=3, batch_size=16)
        return statistics.fmean(c.drift for r in m for c in r.clients)

    libre = drift_moyen(0.0)
    contenu = drift_moyen(10.0)

    # Mesuré : 45,5 % de réduction dans ce régime. Le seuil à 20 % laisse une
    # marge confortable sans rendre le test complaisant.
    assert contenu < libre * 0.8, (
        f"le terme proximal ne contient pas la dérive : {contenu:.4e} "
        f"contre {libre:.4e} sans lui"
    )
