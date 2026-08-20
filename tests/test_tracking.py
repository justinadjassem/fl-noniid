"""Tests du puits MLflow.

Ils tournent sur un magasin de fichiers jetable, jamais sur un serveur : le
module ne fait que lire `MLFLOW_TRACKING_URI`, donc ce qui est vrai d'un
`file://` l'est du `http://mlflow:5000` de Docker. C'est ce qui rend ces
tests exécutables en local et en CI, pas seulement dans le conteneur.
"""

from datetime import datetime, timezone

import mlflow
import pytest
from fastapi.testclient import TestClient

from api.main import app

from contracts.schemas import Algo, RoundMetric, Run, RunConfig, RunStatus
from fl_core.tracking import EXPERIMENT, track

CFG = RunConfig(algo=Algo.fedprox, mu=0.01, alpha=0.1, rounds=3, seed=0)


def _metric(run_id: str, r: int) -> RoundMetric:
    return RoundMetric(
        run_id=run_id, round=r,
        global_acc=0.80 + 0.01 * r, global_loss=1.0 / r,
        mean_client_acc=0.75 + 0.01 * r, std_client_acc=0.05,
        comm_mb=1.43 * r, wall_time_s=0.5 * r,
    )


def _run(run_id: str, **kw) -> Run:
    return Run(
        id=run_id, config=CFG, status=RunStatus.done,
        created_at=datetime.now(timezone.utc), **kw,
    )


def _retrouver(run_id: str):
    """Retrouve le run MLflow par son tag — la clé de jointure de BACK-2."""
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    assert exp is not None, f"expérience {EXPERIMENT!r} jamais créée"
    found = client.search_runs([exp.experiment_id], f"tags.fl_run_id = '{run_id}'")
    assert len(found) == 1, f"{len(found)} run(s) pour {run_id!r}, attendu 1"
    return client, found[0]


@pytest.fixture
def magasin_jetable(tmp_path, monkeypatch):
    """Isole complètement MLflow dans tmp_path, et restaure l'état global.

    Backend sqlite et non magasin de fichiers : MLflow 3.x refuse `file://`
    (« maintenance mode »). C'est aussi ce que sert le conteneur, donc le test
    s'exécute sur le même moteur de stockage que la production.

    L'expérience est créée ici avec un `artifact_location` explicite : sinon
    MLflow la poserait dans un `./mlruns` relatif au répertoire courant, et la
    suite de tests salirait le dépôt.
    """
    avant = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(EXPERIMENT, artifact_location=str(tmp_path / "artifacts"))
    yield tmp_path
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(avant)


def test_sans_uri_le_puits_est_inerte(monkeypatch):
    """Sans MLFLOW_TRACKING_URI, tout continue de tourner sans rien journaliser.

    C'est ce qui garantit que pytest, un `uvicorn --reload` local et
    l'exécution sur Colab ne dépendent pas d'une infrastructure de suivi.
    Un puits qui exigerait un serveur ferait échouer la suite entière.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with track("inerte", CFG) as tr:
        assert mlflow.active_run() is None
        tr.on_round(_metric("inerte", 1))
        tr.summarize(_run("inerte", final_acc=0.9))

    assert mlflow.active_run() is None


def test_les_metriques_de_chaque_round_sont_journalisees(magasin_jetable):
    """Une série indexée par round, pas une valeur finale.

    `step=m.round` est le point clé : c'est ce qui permet à MLflow de tracer
    les courbes et de comparer des runs de longueurs différentes.
    """
    with track("run-metriques", CFG) as tr:
        for r in (1, 2, 3):
            tr.on_round(_metric("run-metriques", r))

    client, run = _retrouver("run-metriques")

    historique = client.get_metric_history(run.info.run_id, "global_acc")
    assert [h.step for h in historique] == [1, 2, 3]
    assert historique[-1].value == pytest.approx(0.83)

    for champ in ("global_loss", "mean_client_acc", "std_client_acc", "comm_mb", "wall_time_s"):
        assert client.get_metric_history(run.info.run_id, champ), f"{champ} jamais journalisé"


def test_la_config_devient_des_parametres_filtrables(magasin_jetable):
    """Sans ça, BACK-2 ne pourrait pas retrouver les runs par alpha ou par algo."""
    with track("run-params", CFG):
        pass

    _, run = _retrouver("run-params")

    assert run.data.params["algo"] == "fedprox"
    assert float(run.data.params["alpha"]) == 0.1
    assert float(run.data.params["mu"]) == 0.01
    assert int(run.data.params["seed"]) == 0


def test_le_resume_final_est_journalise(magasin_jetable):
    """final_acc et rounds_to_target sont de l'algorithme, pas du stockage :
    MLflow ne les calcule pas, c'est à nous de les verser."""
    with track("run-resume", CFG) as tr:
        tr.on_round(_metric("run-resume", 1))
        tr.summarize(_run("run-resume", final_acc=0.934, rounds_to_target=7))

    _, run = _retrouver("run-resume")

    assert run.data.metrics["final_acc"] == pytest.approx(0.934)
    assert run.data.metrics["rounds_to_target"] == 7


def test_une_cible_jamais_atteinte_n_est_pas_journalisee(magasin_jetable):
    """rounds_to_target = None est une information, pas un zéro.

    Le journaliser à 0 ferait croire à une convergence immédiate et
    fausserait toute moyenne calculée dessus.
    """
    with track("run-jamais", CFG) as tr:
        tr.summarize(_run("run-jamais", final_acc=0.42, rounds_to_target=None))

    _, run = _retrouver("run-jamais")

    assert "rounds_to_target" not in run.data.metrics


def test_une_exception_marque_le_run_en_echec(magasin_jetable):
    """Un run qui plante ne doit pas ressembler à un run réussi.

    Sans ça, un entraînement interrompu à la 40e heure sur Colab apparaîtrait
    dans MLflow comme un résultat valide, simplement plus court.
    """
    with pytest.raises(ValueError):
        with track("run-echec", CFG) as tr:
            tr.on_round(_metric("run-echec", 1))
            raise ValueError("boum")

    _, run = _retrouver("run-echec")

    assert run.info.status == "FAILED"


def test_l_api_verse_ses_runs_dans_mlflow(magasin_jetable):
    """La chaîne complète : POST /runs -> moteur -> puits MLflow.

    C'est ce qui rend BACK-1 démontrable. Sans cette couture, le service
    tournerait à vide : on livrerait une UI MLflow vide, sans aucune preuve
    que le puits fonctionne.
    """
    with TestClient(app) as c:
        reponse = c.post(
            "/runs", json={"config": {"algo": "fedavg", "alpha": 0.1, "rounds": 3}}
        )
        assert reponse.status_code == 201
        run_id = reponse.json()["id"]

    client, run = _retrouver(run_id)

    assert run.data.params["algo"] == "fedavg"
    historique = client.get_metric_history(run.info.run_id, "global_acc")
    assert [h.step for h in historique] == [1, 2, 3]
    assert "final_acc" in run.data.metrics


def test_un_run_qui_plante_est_marque_en_echec_dans_mlflow(magasin_jetable, monkeypatch):
    """L'API attrape les exceptions du moteur pour les afficher dans l'UI.

    Effet de bord : l'exception ne traverse plus le contexte MLflow, qui
    clôturerait donc le run en FINISHED. Un entraînement planté à la 40e heure
    sur Colab ressemblerait à un résultat valide, simplement plus court.
    """

    class MoteurQuiPlante:
        def run(self, run_id, cfg, on_round):
            raise RuntimeError("moteur cassé")

    monkeypatch.setattr("api.main.get_runner", lambda: MoteurQuiPlante())

    with TestClient(app) as c:
        run_id = c.post(
            "/runs", json={"config": {"algo": "fedavg", "alpha": 0.1, "rounds": 3}}
        ).json()["id"]

    _, run = _retrouver(run_id)

    assert run.info.status == "FAILED"
