"""Tests de l'API au stade socle.

Le stockage est en mémoire : ces tests devront survivre au passage à
MLflow, puisqu'ils portent sur le contrat HTTP, pas sur le
stockage.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_cycle_de_vie_d_un_run(client):
    r = client.post("/runs", json={"config": {"algo": "fedavg", "alpha": 0.1, "rounds": 10}})
    assert r.status_code == 201
    run_id = r.json()["id"]

    # BackgroundTasks s'exécute à la sortie du contexte TestClient,
    # donc on relit après coup dans un nouveau contexte.
    metrics = client.get(f"/runs/{run_id}/metrics").json()
    assert [m["round"] for m in metrics] == list(range(1, 11))
    assert metrics[-1]["global_acc"] > metrics[0]["global_acc"]


def test_fedprox_sans_mu_est_refuse(client):
    r = client.post("/runs", json={"config": {"algo": "fedprox", "mu": 0, "alpha": 0.1}})
    assert r.status_code == 422


def test_run_inconnu(client):
    assert client.get("/runs/inexistant").status_code == 404
