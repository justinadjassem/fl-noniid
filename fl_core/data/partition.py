"""Partition non-IID par loi de Dirichlet.

C'est la variable indépendante de toute l'étude : tout le reste — FedAvg,
FedProx, les deux bornes — se mesure en fonction de alpha.

Principe : pour chaque classe, on tire un vecteur de proportions dans une loi
de Dirichlet et on répartit les indices de cette classe selon ces proportions.

    alpha petit  (0,05) -> tirages très déséquilibrés, un client rafle presque
                           toute une classe
    alpha grand  (100)  -> tirages proches de l'uniforme, chaque client voit
                           tout : régime quasi-IID

Du numpy pur, pas de torch : on partitionne le vecteur d'étiquettes, jamais les
images. C'est instantané et ça s'itère en secondes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int = 0,
    min_size: int = 10,
    max_draws: int = 100,
) -> tuple[list[np.ndarray], int]:
    """Partitionne les indices entre n_clients selon Dir(alpha) par classe.

    Retourne (liste d'index par client, nombre de tirages effectués).

    Le retirage protège contre les clients vides : aux petits alpha, Dirichlet
    attribue régulièrement zéro échantillon à un client. Sans ce garde-fou, la
    boucle fédérée planterait — ou pire, tournerait avec 9 clients au lieu de
    10 sans rien signaler.
    """
    if n_clients < 2:
        raise ValueError("Il faut au moins 2 clients.")
    if alpha <= 0:
        raise ValueError("alpha doit être strictement positif.")

    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1

    for draw in range(1, max_draws + 1):
        buckets: list[list[int]] = [[] for _ in range(n_clients)]

        for c in range(n_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)

            # proportions de la classe c attribuées à chaque client
            p = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]

            for k, part in enumerate(np.split(idx_c, cuts)):
                buckets[k].extend(part.tolist())

        if min(len(b) for b in buckets) >= min_size:
            return [np.sort(np.array(b, dtype=np.int64)) for b in buckets], draw

    raise RuntimeError(
        f"alpha={alpha} avec {n_clients} clients : impossible d'obtenir "
        f"{min_size} échantillons par client en {max_draws} tirages. "
        f"Réduisez n_clients ou min_size."
    )


def partition_matrix(
    labels: np.ndarray, partitions: list[np.ndarray], n_classes: int = 10
) -> np.ndarray:
    """Matrice clients x classes des effectifs. Alimente la heatmap."""
    m = np.zeros((len(partitions), n_classes), dtype=int)
    for k, idx in enumerate(partitions):
        m[k] = np.bincount(labels[idx], minlength=n_classes)
    return m


def heterogeneity_score(matrix: np.ndarray) -> float:
    """Distance L1 moyenne entre la distribution de chaque client et l'uniforme.

    0 = parfaitement IID ; proche de 2 = chaque client ne voit qu'une classe.

    Permet de rapporter l'hétérogénéité RÉELLEMENT obtenue, pas seulement
    l'alpha demandé : à alpha fixé, un tirage Dirichlet varie beaucoup d'un
    seed à l'autre. C'est une source de variance à documenter dans le rapport.
    """
    props = matrix / matrix.sum(axis=1, keepdims=True)
    uniform = 1.0 / matrix.shape[1]
    return float(np.abs(props - uniform).sum(axis=1).mean())


def save_manifest(
    path: str | Path,
    labels: np.ndarray,
    partitions: list[np.ndarray],
    alpha: float,
    seed: int,
    draws: int,
) -> dict:
    """Manifeste JSON versionnable : la preuve du régime non-IID de ce run."""
    matrix = partition_matrix(labels, partitions)
    payload = {
        "alpha": alpha,
        "seed": seed,
        "n_clients": len(partitions),
        "draws_needed": draws,
        "heterogeneity_score": heterogeneity_score(matrix),
        "sizes": [int(len(p)) for p in partitions],
        "classes_per_client": [int((row > 0).sum()) for row in matrix],
        "class_counts": matrix.tolist(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
