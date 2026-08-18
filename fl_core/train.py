"""Entraînement local et évaluation.

Ces deux fonctions sont utilisées par TOUTES les méthodes comparées : le
centralisé, le local pur, FedAvg, FedProx, et le client Flower. C'est ce qui
rend la comparaison scientifiquement propre — un seul chemin de code, une
seule variable qui change.

Le paramètre `mu` est présent dès maintenant mais reste inactif jusqu'à ML-4 :
c'est lui qui transformera FedAvg en FedProx, sans dupliquer une ligne.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def train_local(
    model: nn.Module,
    loader,
    epochs: int,
    lr: float,
    device: str = "cpu",
    mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
) -> nn.Module:
    """Entraîne le modèle sur les données locales.

    `mu > 0` active le terme proximal de FedProx :

        min_w  F_k(w) + (mu/2) * ||w - w_global||^2

    ATTENTION — `global_params` doit être capturé UNE SEULE FOIS au début du
    round, jamais rafraîchi entre les batches. Sinon le terme s'annule et
    FedProx redevient FedAvg sans qu'aucune erreur ne le signale.
    """
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            loss = crit(model(x), y)

            if mu > 0.0 and global_params is not None:
                prox = sum(
                    ((p - g) ** 2).sum()
                    for p, g in zip(model.parameters(), global_params)
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            opt.step()

    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str = "cpu") -> tuple[float, float]:
    """Retourne (accuracy, loss moyenne) sur le jeu fourni."""
    model.to(device).eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += crit(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()

    return correct / n, total_loss / n


def model_size_mb(model: nn.Module) -> float:
    """Volume échangé par un client à chaque round. Alimente comm_mb."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
