"""API du système : orchestration des expériences.

ÉTAT SOCLE — le stockage est en mémoire.
MLflow le remplacera
le but est de prouver que la chaîne contrat → moteur → API → dashboard
fonctionne de bout en bout.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query

from contracts.schemas import (
    Algo,
    CreateRunRequest,
    RoundMetric,
    Run,
    RunStatus,
)
from fl_core.runner import get_runner

app = FastAPI(
    title="FL non-IID — FedAvg vs FedProx",
    version="0.1.0",
    description="Orchestration des expériences d'apprentissage fédéré non-IID.",
)

_RUNS: dict[str, Run] = {}
_METRICS: dict[str, list[RoundMetric]] = defaultdict(list)


def _execute(run_id: str) -> None:
    """Exécute une expérience en tâche de fond."""
    run = _RUNS[run_id]
    run.status = RunStatus.running

    def sink(m: RoundMetric) -> None:
        _METRICS[run_id].append(m)
        run.current_round = m.round

    try:
        get_runner().run(run_id, run.config, sink)

        accs = [m.global_acc for m in _METRICS[run_id]]
        tail = accs[-5:] or accs
        run.final_acc = sum(tail) / len(tail)
        run.rounds_to_target = next(
            (m.round for m in _METRICS[run_id]
             if m.global_acc >= run.config.target_acc),
            None,
        )
        run.status = RunStatus.done
    except Exception as exc:                      # noqa: BLE001 — tracer toute panne dans l'UI
        run.status = RunStatus.failed
        run.error = f"{type(exc).__name__}: {exc}"
    finally:
        run.finished_at = datetime.now(timezone.utc)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "runner": type(get_runner()).__name__}


@app.post("/runs", response_model=Run, status_code=201)
def create_run(req: CreateRunRequest, background: BackgroundTasks) -> Run:
    cfg = req.config

    if cfg.algo is Algo.fedprox and cfg.mu <= 0:
        raise HTTPException(422, "FedProx exige mu > 0 (sinon c'est exactement FedAvg).")
    if cfg.algo is not Algo.fedprox and cfg.mu != 0:
        raise HTTPException(422, f"mu doit valoir 0 pour l'algorithme {cfg.algo.value}.")

    run_id = uuid.uuid4().hex[:12]
    run = Run(
        id=run_id,
        config=cfg,
        status=RunStatus.pending,
        created_at=datetime.now(timezone.utc),
    )
    _RUNS[run_id] = run
    background.add_task(_execute, run_id)
    return run


@app.get("/runs", response_model=list[Run])
def list_runs() -> list[Run]:
    return sorted(_RUNS.values(), key=lambda r: r.created_at, reverse=True)


@app.get("/runs/{run_id}", response_model=Run)
def get_run(run_id: str) -> Run:
    if run_id not in _RUNS:
        raise HTTPException(404, "Run inconnu.")
    return _RUNS[run_id]


@app.get("/runs/{run_id}/metrics", response_model=list[RoundMetric])
def get_metrics(run_id: str, since_round: int = Query(0, ge=0)) -> list[RoundMetric]:
    if run_id not in _RUNS:
        raise HTTPException(404, "Run inconnu.")
    return [m for m in _METRICS[run_id] if m.round > since_round]
