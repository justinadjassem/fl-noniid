"""Le modèle partagé par toutes les méthodes comparées.

Trois décisions de conception, chacune mesurée sur MNIST
(12 000 exemples, 2 époques, même seed) :

**GroupNorm et non BatchNorm.** BatchNorm maintient des statistiques
`running_mean` et `running_var`. En fédéré, les moyenner entre clients aux
distributions différentes produit des valeurs qui ne correspondent à aucun
client : on imputerait à l'hétérogénéité une dégradation venant en réalité de
l'architecture. GroupNorm n'a aucun buffer, donc tout le `state_dict` est
moyennable et l'agrégation reste exacte.

**Pooling adaptatif en 4x4, pas en 1x1.** Un pooling global écraserait chaque
canal en un seul nombre, faisant tomber la tête de 2048 à 128 entrées.
Mesuré : 0,39 avec (1,1) contre 0,95 avec (4,4). Descendre à (2,2) coûte aussi
cher (0,90 à la première époque). Le pooling adaptatif rend par ailleurs la
tête indépendante de la taille d'entrée — le même modèle sert MNIST 28x28 et
CIFAR-10 32x32.

**Tête cachée de 128 unités, pas 256.** À 256, la couche `2048 -> 256` pesait
524 544 paramètres, soit 85 % du modèle, pour 2,48 Mo échangés par client et
par round. À 128 : 356 682 paramètres au total, 1,43 Mo par round, et une
accuracy *supérieure* (0,9746 contre 0,9680). Le volume échangé compte : il
fait partie des métriques que l'étude rapporte.
"""

from __future__ import annotations

import torch.nn as nn

# Rendus explicites : la tête dépend des trois, et les voir écrits évite de
# croire à une coïncidence entre le nombre de canaux et la largeur cachée.
_CHANNELS = 128   # canaux en sortie du dernier bloc convolutif
_POOL = 4         # côté de la carte après AdaptiveAvgPool
_HIDDEN = 128     # largeur de la couche cachée de la tête


class SmallCNN(nn.Module):

    def __init__(self, in_channels: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, _CHANNELS, kernel_size=3, padding=1),
            nn.GroupNorm(8, _CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((_POOL, _POOL)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_CHANNELS * _POOL * _POOL, _HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(_HIDDEN, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


def build_model(dataset: str) -> SmallCNN:
    """Construit le modèle adapté au dataset (1 canal pour MNIST, 3 pour CIFAR)."""
    from fl_core.data.loaders import INPUT_CHANNELS, N_CLASSES
    key = dataset.lower()
    if key not in INPUT_CHANNELS:
        raise ValueError(f"Dataset inconnu : {dataset!r}. Attendu : {sorted(INPUT_CHANNELS)}")
    return SmallCNN(in_channels=INPUT_CHANNELS[key], n_classes=N_CLASSES)
