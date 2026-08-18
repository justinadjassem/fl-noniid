"""Déterminisme.

Sans ces quelques lignes, les trois seeds exigés par le protocole ne veulent
rien dire : deux exécutions d'une même configuration donneraient des résultats
différents, et l'écart entre FedAvg et FedProx deviendrait indéfendable.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fixe les quatre sources d'aléa du projet.

    - `random`  : tirage du nombre d'époques locales (hétérogénéité systèmes)
    - `numpy`   : partition de Dirichlet, sélection des clients par round
    - `torch`   : initialisation des poids, ordre des batches
    - `cudnn`   : algorithmes non déterministes sur GPU (sans effet sur CPU,
                  mais nécessaire si la grille finale tourne sur Colab)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
