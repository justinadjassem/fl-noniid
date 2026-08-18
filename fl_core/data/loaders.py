"""Chargement des datasets. Aucune logique fédérée ici.

Les données brutes vivent dans `data/` (ignoré par git) : torchvision les
retélécharge si besoin. Seul le manifeste de partition est versionné.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

# Statistiques de normalisation standard de la littérature.
_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

INPUT_CHANNELS = {"mnist": 1, "cifar10": 3}
N_CLASSES = 10


@dataclass(frozen=True)
class Split:
    train: Dataset
    test: Dataset
    labels: np.ndarray  # étiquettes du train, pour partitionner sans charger les images


def _transform(name: str, augment: bool = False):
    mean, std = _STATS[name]
    steps = []
    if augment and name == "cifar10":
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


def load(name: str, augment: bool = False, download: bool = True) -> Split:
    """Retourne les splits train/test et le vecteur d'étiquettes du train."""
    name = name.lower()
    if name not in _STATS:
        raise ValueError(f"Dataset inconnu : {name!r}. Attendu : {sorted(_STATS)}")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    cls = datasets.MNIST if name == "mnist" else datasets.CIFAR10

    train = cls(DATA_ROOT, train=True, download=download, transform=_transform(name, augment))
    test = cls(DATA_ROOT, train=False, download=download, transform=_transform(name))

    # MNIST expose `targets` sous forme de tenseur, CIFAR-10 sous forme de liste.
    labels = np.asarray(train.targets, dtype=np.int64)
    return Split(train=train, test=test, labels=labels)


if __name__ == "__main__":  # pragma: no cover - analyse exploratoire
    import sys

    for ds in sys.argv[1:] or ["mnist", "cifar10"]:
        split = load(ds)
        counts = np.bincount(split.labels, minlength=N_CLASSES)
        print(f"\n=== {ds} ===")
        print(f"train {len(split.train)}  test {len(split.test)}")
        print(f"effectif par classe : {counts.tolist()}")
        print(f"min {counts.min()}  max {counts.max()}  "
              f"déséquilibre {counts.max() / counts.min():.3f}")
