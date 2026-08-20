# Apprentissage fédéré sur données non-IID — FedAvg vs FedProx

Projet Annuel AI & Big Data.
Comparaison de **FedAvg** et **FedProx** sur des données partitionnées de façon
non-IID (partition de Dirichlet), encadrée par deux bornes : entraînement
centralisé (borne haute) et entraînement purement local (borne basse).

## Démarrage

```bash
docker compose up --build
```

Dashboard : http://localhost:8501 · API : http://localhost:8000/docs ·
Suivi MLflow : http://localhost:5000

> Le port 5000 est souvent déjà pris (AirPlay sur macOS, une autre stack de
> données sous Linux). Dans ce cas :
> `MLFLOW_PORT=5001 docker compose up --build`. Seul le port côté hôte change ;
> l'API joint toujours le service sur `http://mlflow:5000` en interne.

> **Prérequis : Docker Compose v2**, la commande en deux mots. L'ancien binaire
> `docker-compose` v1 est abandonné depuis 2023 et casse avec les moteurs Docker
> récents (`KeyError: 'ContainerConfig'` dès qu'un conteneur doit être recréé).
> Sous Ubuntu : `sudo apt install docker-compose-v2`.

## Sans Docker

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate         # Windows

# Sous Linux, `pip install torch` tire la build CUDA (~2,5 Go) dont le projet
# n'a aucun usage : installer la build CPU AVANT le reste.
pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.10.0 torchvision==0.25.0

pip install -r requirements/dev.txt
pip install -e .        # sans ça : ModuleNotFoundError: No module named 'contracts'

uvicorn api.main:app --reload      # terminal 1
streamlit run app/dashboard.py     # terminal 2
```

Hors Docker, `MLFLOW_TRACKING_URI` est absent : le suivi d'expériences est
alors inerte et rien n'est journalisé. C'est voulu — le cœur scientifique ne
dépend d'aucune infrastructure de suivi.

## Tests

```bash
pytest -q
```