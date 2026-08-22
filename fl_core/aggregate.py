"""Agrégation FedAvg : moyenne pondérée par la taille du jeu local.

C'est la seule opération que le serveur exécute lui-même — tout le reste est
de l'entraînement local. La pondération n'est pas cosmétique : l'objectif
global est

    F(w) = somme_k (n_k / n) * F_k(w)

et la moyenne pondérée est l'agrégation cohérente avec cet objectif. Aux
petits alpha, les tailles de clients varient d'un facteur 50 : une moyenne
uniforme donnerait autant de poids au client de 80 images qu'à celui de 4 000.

Avec GroupNorm, tous les tenseurs du `state_dict` sont des paramètres
apprenables flottants — pas de compteur entier ni de statistique de batch à
traiter à part. C'est le bénéfice concret du choix d'architecture.
"""

from __future__ import annotations

import copy

import torch


def average_weights(
    state_dicts: list[dict[str, torch.Tensor]],
    n_samples: list[int],
) -> dict[str, torch.Tensor]:
    """w = somme_k (n_k / n) * w_k

    `state_dicts` n'est jamais modifié : le premier sert de gabarit et est
    copié. Écrire dedans corromprait silencieusement le broadcast du round
    suivant.
    """
    if not state_dicts:
        raise ValueError("Aucun state_dict à agréger.")
    if len(state_dicts) != len(n_samples):
        raise ValueError(
            f"{len(state_dicts)} state_dicts pour {len(n_samples)} tailles."
        )

    total = sum(n_samples)
    if total <= 0:
        raise ValueError("La somme des tailles de shards doit être positive.")

    avg = copy.deepcopy(state_dicts[0])

    for key in avg:
        avg[key] = sum(
            sd[key].to(torch.float32) * (n / total)
            for sd, n in zip(state_dicts, n_samples)
        )
    return avg
