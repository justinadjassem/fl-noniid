"""Tests de la partition de Dirichlet.

Ils portent sur les propriétés mathématiques attendues, pas sur des valeurs
figées : la partition doit rester exhaustive, disjointe, reproductible, et
produire une hétérogénéité qui croît quand alpha diminue.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl_core.data.partition import (
    dirichlet_partition,
    heterogeneity_score,
    partition_matrix,
)

# 6000 exemples, 10 classes parfaitement équilibrées : toute hétérogénéité
# observée vient donc de la partition, jamais du jeu de départ.
LABELS = np.repeat(np.arange(10), 600)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.5, 1.0, 100.0])
def test_partition_exhaustive_et_disjointe(alpha: float):
    parts, _ = dirichlet_partition(LABELS, 10, alpha, seed=0)
    tous = np.concatenate(parts)
    assert len(tous) == len(LABELS)              # rien perdu
    assert len(np.unique(tous)) == len(LABELS)   # rien dupliqué


def test_reproductible_a_seed_egale():
    a, _ = dirichlet_partition(LABELS, 10, 0.1, seed=42)
    b, _ = dirichlet_partition(LABELS, 10, 0.1, seed=42)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_seeds_differents_donnent_des_partitions_differentes():
    a, _ = dirichlet_partition(LABELS, 10, 0.1, seed=0)
    b, _ = dirichlet_partition(LABELS, 10, 0.1, seed=1)
    assert not all(np.array_equal(x, y) for x, y in zip(a, b))


def test_aucun_client_vide():
    parts, _ = dirichlet_partition(LABELS, 10, 0.05, seed=0, min_size=10)
    assert min(len(p) for p in parts) >= 10


def test_alpha_grand_donne_du_quasi_iid():
    """À alpha=100, chaque client doit voir les dix classes."""
    parts, _ = dirichlet_partition(LABELS, 10, 100.0, seed=0)
    for p in parts:
        assert len(np.unique(LABELS[p])) == 10


def test_alpha_petit_concentre_les_classes():
    """À alpha=0.05, au moins un client doit être très spécialisé."""
    parts, _ = dirichlet_partition(LABELS, 10, 0.05, seed=0)
    assert min(len(np.unique(LABELS[p])) for p in parts) <= 3


def test_score_croit_quand_alpha_diminue():
    """La mesure d'hétérogénéité doit être monotone en alpha."""
    scores = [
        heterogeneity_score(
            partition_matrix(LABELS, dirichlet_partition(LABELS, 10, a, seed=0)[0])
        )
        for a in (100.0, 1.0, 0.1, 0.05)
    ]
    assert scores == sorted(scores)


def test_contrainte_infaisable_leve_une_erreur():
    """Mieux vaut un message explicite qu'un plantage obscur plus loin."""
    with pytest.raises(RuntimeError, match="impossible d'obtenir"):
        dirichlet_partition(LABELS, 10, 0.001, seed=0, min_size=500, max_draws=5)


def test_parametres_invalides():
    with pytest.raises(ValueError):
        dirichlet_partition(LABELS, 1, 0.5)
    with pytest.raises(ValueError):
        dirichlet_partition(LABELS, 10, 0.0)
