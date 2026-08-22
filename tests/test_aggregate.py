"""Tests de l'agrégation FedAvg.

C'est la seule opération que le serveur exécute lui-même, et une erreur ici
produit des courbes plausibles mais fausses : le modèle converge quand même,
plus lentement, et rien ne signale que la pondération est cassée. D'où des
tests sur des valeurs calculables à la main plutôt que sur des propriétés
vagues.
"""

import copy

import torch

from fl_core.aggregate import average_weights
from fl_core.models.cnn import build_model


def _sd(valeur: float) -> dict[str, torch.Tensor]:
    """Un state_dict minimal, pour raisonner sur des nombres lisibles."""
    return {"w": torch.full((2, 2), valeur), "b": torch.tensor([valeur])}


def test_moyenne_ponderee_exacte():
    """w = somme_k (n_k / n) * w_k, sur un cas calculable de tête.

    3 exemples à 1.0 et 1 exemple à 5.0 -> (3*1 + 1*5) / 4 = 2.0
    """
    avg = average_weights([_sd(1.0), _sd(5.0)], [3, 1])

    assert torch.allclose(avg["w"], torch.full((2, 2), 2.0))
    assert torch.allclose(avg["b"], torch.tensor([2.0]))


def test_la_ponderation_est_reellement_utilisee():
    """Sans ce test, une moyenne simple passerait le précédent par hasard
    si les tailles étaient égales. Ici la moyenne simple donnerait 3.0.

    L'enjeu est réel : aux petits alpha, les tailles de clients varient d'un
    facteur 50, et une moyenne uniforme sur-représenterait massivement le
    client à 80 images.
    """
    avg = average_weights([_sd(1.0), _sd(5.0)], [3, 1])

    moyenne_simple = 3.0
    assert not torch.allclose(avg["w"], torch.full((2, 2), moyenne_simple))


def test_un_seul_client_rend_ses_propres_poids():
    avg = average_weights([_sd(4.2)], [17])

    assert torch.allclose(avg["w"], torch.full((2, 2), 4.2))


def test_les_entrees_ne_sont_pas_modifiees():
    """L'agrégation doit être pure.

    Si elle écrivait dans le state_dict du premier client, le broadcast du
    round suivant partirait d'un modèle silencieusement corrompu.
    """
    a, b = _sd(1.0), _sd(5.0)
    avant = copy.deepcopy(a)

    average_weights([a, b], [3, 1])

    assert torch.allclose(a["w"], avant["w"])
    assert torch.allclose(a["b"], avant["b"])


def test_le_state_dict_du_vrai_modele_passe_entier():
    """Aucune clé perdue en route.

    C'est le bénéfice concret de GroupNorm : tous les tenseurs sont des
    paramètres apprenables flottants, donc la même formule s'applique à
    l'intégralité du state_dict, sans compteur entier à traiter à part.
    """
    a, b = build_model("mnist").state_dict(), build_model("mnist").state_dict()

    avg = average_weights([a, b], [100, 300])

    assert set(avg) == set(a)
    for cle in avg:
        assert avg[cle].shape == a[cle].shape
        assert avg[cle].dtype.is_floating_point, f"{cle} n'est pas flottant"
