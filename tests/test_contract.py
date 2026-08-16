"""Tests de la couture : le moteur respecte-t-il le contrat ?

Quand FlowerRunner remplacera FakeRunner, ces tests doivent continuer
à passer sans modification. C'est ce qui en fait des tests de non-régression.
"""

import pytest

from contracts.schemas import Algo, RoundMetric, RunConfig
from fl_core.runner import FakeRunner


def collect(cfg: RunConfig) -> list[RoundMetric]:
    out: list[RoundMetric] = []
    FakeRunner(seconds_per_round=0.0).run("test", cfg, out.append)
    return out


def test_un_metric_par_round():
    m = collect(RunConfig(algo=Algo.fedavg, alpha=0.5, rounds=20))
    assert [x.round for x in m] == list(range(1, 21))


def test_accuracy_dans_les_bornes():
    m = collect(RunConfig(algo=Algo.fedavg, alpha=0.1, rounds=30))
    assert all(0.0 <= x.global_acc <= 1.0 for x in m)


def test_reproductible_a_seed_egale():
    cfg = RunConfig(algo=Algo.fedavg, alpha=0.1, rounds=15, seed=7)
    assert [x.global_acc for x in collect(cfg)] == [x.global_acc for x in collect(cfg)]


def test_le_centralise_domine_le_local():
    common = dict(alpha=0.1, rounds=40, seed=0)
    assert (collect(RunConfig(algo=Algo.centralized, **common))[-1].global_acc
            > collect(RunConfig(algo=Algo.local, **common))[-1].global_acc)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.5])
def test_fedprox_aide_en_forte_heterogeneite(alpha):
    common = dict(alpha=alpha, rounds=80, seed=0)
    assert (collect(RunConfig(algo=Algo.fedprox, mu=0.01, **common))[-1].global_acc
            >= collect(RunConfig(algo=Algo.fedavg, **common))[-1].global_acc)


def test_le_drift_croit_quand_alpha_diminue():
    fort = collect(RunConfig(algo=Algo.fedavg, alpha=0.05, rounds=30))
    faible = collect(RunConfig(algo=Algo.fedavg, alpha=100.0, rounds=30))
    assert fort[-1].std_client_acc > faible[-1].std_client_acc
