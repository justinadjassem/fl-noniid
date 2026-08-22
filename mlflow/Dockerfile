# Serveur de suivi d'expériences.
#
# Image volontairement mince : ce service n'exécute ni torch, ni Flower, ni le
# code du projet. Il ne fait que stocker des métriques et les afficher. Copier
# fl_core/ ici ferait passer l'image de ~250 Mo à plus de 2 Go pour rien.
FROM python:3.12-slim

# Même version que requirements/core.txt : le client et le serveur doivent
# parler le même protocole.
RUN pip install --no-cache-dir mlflow==3.15.1

WORKDIR /mlruns
EXPOSE 5000

# --backend-store-uri en sqlite, PAS en magasin de fichiers : MLflow 3.x
# refuse `./mlruns` (« maintenance mode ») et exige un backend SQLAlchemy.
#
# Quatre barres obliques dans `sqlite:////mlruns/...` : trois pour le schéma
# d'URI, la quatrième est la racine du chemin absolu.
#
# /mlruns est monté depuis ./mlruns sur l'hôte : la base survit à un
# `docker compose down`, et c'est ce fichier que la grille finale versionnera.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlruns/mlflow.db", \
     "--artifacts-destination", "/mlruns/artifacts", \
     "--serve-artifacts"]
