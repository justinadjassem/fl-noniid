"""Les deux bornes qui encadrent toute l'étude.

    centralisé  -> borne HAUTE : toutes les données réunies, l'idéal
                   inatteignable en fédéré
    local pur   -> borne BASSE : chaque client s'entraîne seul, aucune
                   agrégation. Répond à « à quoi bon fédérer ? »

Les deux émettent des RoundMetric conformes au contrat, donc le dashboard les
affiche sans une ligne de code spécifique.
"""

from __future__ import annotations

import statistics
import time

from torch.utils.data import DataLoader, Subset

from contracts.schemas import RoundMetric, RunConfig
from fl_core.data.loaders import load
from fl_core.data.partition import dirichlet_partition
from fl_core.models.cnn import build_model
from fl_core.seeding import seed_everything
from fl_core.train import evaluate, train_local


def run_centralized(run_id: str, cfg: RunConfig, on_round):
    """Borne haute. Une époque sur l'ensemble des données = un « round ».

    Pas de communication : comm_mb reste à zéro, et c'est précisément le point
    — cette borne ignore la contrainte que le fédéré doit respecter.
    """
    seed_everything(cfg.seed)
    split = load(cfg.dataset.value)
    model = build_model(cfg.dataset.value)

    train_loader = DataLoader(split.train, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(split.test, batch_size=512)

    t0 = time.time()
    for r in range(1, cfg.rounds + 1):
        train_local(model, train_loader, epochs=1, lr=cfg.lr)
        acc, loss = evaluate(model, test_loader)
        on_round(RoundMetric(
            run_id=run_id, round=r,
            global_acc=acc, global_loss=loss,
            mean_client_acc=acc, std_client_acc=0.0,
            comm_mb=0.0,
            wall_time_s=round(time.time() - t0, 2),
        ))
    return model


def run_local_only(run_id: str, cfg: RunConfig, on_round):
    """Borne basse. Chaque client s'entraîne seul, aucune agrégation.

    Chaque modèle local est évalué sur le jeu de test GLOBAL — seul choix
    comparable à FedAvg et au centralisé. Un client évalué sur ses propres
    classes afficherait 95 % et l'on n'en conclurait rien.

    On rapporte la moyenne des N accuracies ; leur écart-type mesure à quel
    point les clients divergent, ce qui est le client drift à son maximum
    (aucune agrégation ne vient le corriger).
    """
    seed_everything(cfg.seed)
    split = load(cfg.dataset.value)
    parts, draws = dirichlet_partition(
        split.labels, cfg.n_clients, cfg.alpha, seed=cfg.seed
    )
    test_loader = DataLoader(split.test, batch_size=512)

    models = [build_model(cfg.dataset.value) for _ in range(cfg.n_clients)]
    loaders = [
        DataLoader(Subset(split.train, idx), batch_size=cfg.batch_size, shuffle=True)
        for idx in parts
    ]

    t0 = time.time()
    for r in range(1, cfg.rounds + 1):
        accs = []
        for model, loader in zip(models, loaders):
            train_local(model, loader, epochs=1, lr=cfg.lr)
            acc, _ = evaluate(model, test_loader)
            accs.append(acc)

        on_round(RoundMetric(
            run_id=run_id, round=r,
            global_acc=statistics.fmean(accs),
            global_loss=0.0,
            mean_client_acc=statistics.fmean(accs),
            std_client_acc=statistics.stdev(accs) if len(accs) > 1 else 0.0,
            comm_mb=0.0,           # aucune communication : c'est le point
            wall_time_s=round(time.time() - t0, 2),
        ))
    return models
