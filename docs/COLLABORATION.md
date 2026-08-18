# Guide de reprise du projet

Tout ce qu'il faut pour prendre le projet en main et le mener à terme, même
sans personne pour l'expliquer.

---

## 1. Le projet en deux minutes

**Question posée par le sujet :** en apprentissage fédéré, quand chaque client
possède des données de distribution différente (**non-IID**), la moyenne des
modèles locaux calculée par **FedAvg** dégrade le modèle global — c'est le
*client drift*. **FedProx** ajoute une pénalité qui empêche chaque modèle local
de trop s'éloigner du modèle global.

> **À partir de quel niveau d'hétérogénéité FedProx surpasse-t-il
> *significativement* FedAvg ?**

Quatre méthodes comparées :

| Méthode | Ce qu'elle fait | Rôle |
|---|---|---|
| **Centralisé** | toutes les données réunies | borne HAUTE |
| **FedAvg** | moyenne pondérée des poids locaux | la référence |
| **FedProx** | FedAvg + terme `(μ/2)·‖w − w_global‖²` | l'objet de l'étude |
| **Local pur** | chaque client seul, aucune agrégation | borne BASSE |

L'hétérogénéité est pilotée par **α** (partition de Dirichlet) : α petit = très
non-IID, α ≥ 100 ≈ IID.

**Papiers de référence**
- McMahan et al. 2017, FedAvg — [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
- Li et al. 2020, FedProx — [arXiv:1812.06127](https://arxiv.org/abs/1812.06127)

---

## 2. Installation

```bash
git clone https://github.com/justinadjassem/fl-noniid.git
cd fl-noniid

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements/dev.txt
pip install -e .                # INDISPENSABLE — voir ci-dessous
```

> ⚠️ **`pip install -e .` n'est pas optionnel.** `streamlit run app/dashboard.py`
> place le dossier du script sur `sys.path`, pas la racine du projet : sans
> installation éditable, tu auras `ModuleNotFoundError: No module named
> 'contracts'`. Ça a déjà coûté une demi-heure à quelqu'un.

**Vérifier que tout marche :**

```bash
pytest -q                       # doit afficher : 31 passed
docker compose up --build       # dashboard sur :8501, API sur :8000/docs
```

Le premier build Docker prend 5 à 10 min (torch CPU + Ray). Les suivants sont
quasi instantanés.

> Les datasets (404 Mo) sont téléchargés automatiquement par torchvision au
> premier usage, dans `data/`. Ils ne sont **jamais** versionnés.

---

## 3. Architecture

```
contracts/schemas.py   LE CONTRAT — RunConfig, RoundMetric, Run, GridCell
                       Modifié à trois, jamais unilatéralement.

fl_core/               Cœur scientifique. Ne connaît ni FastAPI ni Streamlit.
  data/loaders.py      chargement MNIST / CIFAR-10
  data/partition.py    partition de Dirichlet
  models/cnn.py        le CNN (GroupNorm)
  seeding.py           déterminisme
  train.py             train_local() et evaluate() — utilisés par TOUT
  baselines.py         centralisé et local pur
  runner.py            protocole Runner + moteur factice + get_runner()

api/main.py            FastAPI : /health, /runs, /runs/{id}/metrics
app/dashboard.py       Streamlit : convergence, tableau croisé, client drift
scripts/               scripts de vérification
tests/                 31 tests
```

### La couture à comprendre avant tout

`fl_core/runner.py` définit un protocole :

```python
class Runner(Protocol):
    def run(self, run_id: str, cfg: RunConfig, on_round: OnRound) -> None: ...
```

Le moteur **ne sait pas où vont ses métriques** : il appelle `on_round(metric)`.
C'est ce qui permet de brancher, sans toucher au code scientifique, une base
locale, MLflow, ou une API distante depuis Colab. Et c'est ce qui fait que
remplacer le moteur factice par Flower ne changera qu'**une ligne** dans
`get_runner()`.

---

## 4. Comment on travaille ensemble

### Les territoires

| Profil | Dossiers | Ne touche jamais |
|---|---|---|
| **ML** | `fl_core/`, `experiments/`, `scripts/` | `api/`, `app/` |
| **BACK** | `api/`, `mlflow/`, Docker | `fl_core/`*, `app/` |
| **FRONT** | `app/`, `.streamlit/` | `fl_core/`, `api/` |
| *partagé* | `contracts/`, `docs/`, `README.md` | protocole ci-dessous |

\* Exception : `fl_core/tracking.py` (puits MLflow) est écrit par BACK.

### Les branches

```
ml/<livrable>      api/<livrable>      app/<livrable>
```

Une branche = **un livrable**, une à deux séances, fusionnée dès qu'elle marche.
Jamais de branche permanente par personne : cinq séances de divergence, c'est
une séance de fusion en fin de projet, quand il n'y a plus de marge.

```bash
git switch main && git pull
git switch -c ml/fedavg-manuel
# ... travail ...
pytest -q
git add fl_core/ tests/
git commit -m "feat(ml): ..."
git push -u origin ml/fedavg-manuel
```

Puis PR sur GitHub, ou fusion directe si vous êtes seul :

```bash
git switch main
git merge --no-ff ml/fedavg-manuel
git push
```

### Convention de commit

```
<type>(<scope>): <résumé à l'impératif, en anglais>

<corps : POURQUOI, pas quoi. Les décisions et leurs mesures.>
```

Types : `feat`, `fix`, `build`, `docs`, `test`, `chore`.
Scopes : `contracts`, `core`, `ml`, `api`, `app`, `docker`.

**L'historique git est une pièce notée.** Les consignes exigent une répartition
du travail détaillée, sous peine de pénalité. `git shortlog -sn` et la liste
des branches fusionnées la démontrent — chacun commite sous son propre nom.

### Le protocole `contracts/`

Seul fichier dont dépendent les trois couches.

1. On en parle avant
2. Branche `contract/…`, fusionnée **le jour même**
3. Les autres rebasent immédiatement : `git switch <ma-branche> && git rebase main`

| Changement | Impact |
|---|---|
| Ajouter un champ **avec valeur par défaut** | indolore |
| Ajouter un champ obligatoire, renommer, supprimer | casse tous les appelants |

### La règle d'or

> **`main` doit toujours marcher.** À tout instant : clone, une commande,
> l'application s'ouvre. C'est aussi le critère de notation.

---

## 5. Ce qui est fait

| Brique | État | Détail |
|---|---|---|
| Contrat partagé | ✅ | `RunConfig`, `RoundMetric`, `Run`, `GridCell` |
| Moteur factice + couture | ✅ | courbes analytiques, 8 tests |
| API FastAPI | ✅ | stockage en mémoire, 4 tests |
| Dashboard Streamlit | ✅ | 3 onglets, thème violet |
| Docker | ✅ | `docker compose up --build`, 2 services |
| Dépendances en couches | ✅ | `requirements/` + `pyproject.toml` |
| **ML-1** partition Dirichlet | ✅ | 9 tests, heatmap, manifestes JSON |
| **ML-2** modèle et bornes | 🔄 | `cnn.py` fait, bornes en cours |

**31 tests passent.**

### Résultats déjà mesurés

Partition de Dirichlet sur MNIST, 10 clients, seed 0 :

| α | score d'hétérogénéité | classes/client | plus petit client |
|---|---|---|---|
| 0.05 | 1,467 | 3 – 8 | **80 images** |
| 0.1 | 1,359 | 4 – 10 | 1 973 |
| 0.5 | 0,940 | 8 – 10 | 2 204 |
| 100 | 0,079 | 10 – 10 | 5 792 |

Le déséquilibre de **quantité** apparaît gratuitement : à α=0.05 un client n'a
que 80 images quand un autre en a des milliers. C'est le second visage du
non-IID, à mentionner au rapport.

Modèle : **356 682 paramètres** (MNIST), **1,43 Mo échangés par client et par
round**. Ce chiffre alimente `comm_mb`.

---

## 6. Les tâches restantes

### ML-3 · FedAvg à la main
`ml/fedavg-manuel`

**Pourquoi à la main avant Flower :** vous obtenez une implémentation de
référence. Au portage sur Flower, vous comparez les chiffres. Sans elle, un
mauvais résultat aurait trois causes possibles — modèle, partition, config
Flower — sans moyen de les séparer.

**`fl_core/aggregate.py`**

```python
"""Agrégation FedAvg : moyenne pondérée par la taille du jeu local."""

from __future__ import annotations

import copy

import torch


def average_weights(state_dicts, n_samples):
    """w = somme_k (n_k / n) * w_k"""
    total = sum(n_samples)
    avg = copy.deepcopy(state_dicts[0])

    for key in avg:
        avg[key] = sum(
            sd[key].to(torch.float32) * (n / total)
            for sd, n in zip(state_dicts, n_samples)
        )
    return avg
```

Avec GroupNorm, tous les tenseurs du `state_dict` sont des flottants
apprenables — pas de compteur entier à traiter à part. C'est le bénéfice
concret du choix d'architecture.

**`fl_core/server.py`** — la boucle fédérée :

```
seed_everything(cfg.seed)
partitionner les données
créer le modèle global

pour chaque round :
    sélectionner K = round(n_clients * participation) clients
    global_params = [p.detach().clone() for p in global_model.parameters()]   # UNE FOIS

    pour chaque client sélectionné :
        local = copy.deepcopy(global_model)          # broadcast
        epochs_k = tirage dans [1, E] si systems_heterogeneity sinon E
        train_local(local, loader_k, epochs_k, lr, mu=cfg.mu, global_params=global_params)
        évaluer local sur le test global  -> accuracy locale
        collecter (state_dict, taille du shard)

    global_model.load_state_dict(average_weights(states, sizes))
    évaluer le global
    on_round(RoundMetric(...))
```

`std_client_acc` = écart-type des accuracies locales. **C'est la mesure directe
du client drift** — un graphique de plus pour le rapport, gratuit.

**Vérification — commencer en quasi-IID :**

```bash
python scripts/check_federated.py --algo fedavg --alpha 100 --clients 2 --rounds 20
```

> 🛑 **Point d'arrêt.** En quasi-IID, FedAvg doit atteindre **98-99 %** en
> 20-30 rounds. Si vous restez sous 90 %, le bug est dans l'agrégation ou le
> broadcast, **pas** dans l'hétérogénéité. C'est le test qui sépare un bug d'un
> phénomène : à α=0.1 vous ne sauriez jamais lequel des deux vous observez.

Puis allumer l'hétérogénéité :

```bash
for a in 100 1.0 0.5 0.1 0.05; do
  python scripts/check_federated.py --algo fedavg --alpha $a --clients 10 --rounds 60
done
```

Attendu : décroissance **monotone** de l'accuracy quand α diminue, et des
courbes de plus en plus bruitées.

| α | ordre de grandeur, MNIST, 60 rounds |
|---|---|
| 100 | 98 – 99 %, courbe lisse |
| 0.5 | 93 – 97 % |
| 0.1 | 85 – 94 %, oscillations |
| 0.05 | 70 – 90 %, très instable |

---

### ML-4 · FedProx
`ml/fedprox`

**Le code est déjà écrit** — c'est le bloc `if mu > 0.0` de `train_local()`.
Il suffit de passer `mu > 0` dans la `RunConfig`. FedAvg et FedProx partagent
strictement le même chemin de code, ce qui rend la comparaison propre : à
affirmer dans le rapport.

**`tests/test_fedprox.py`** — deux tests, et ce sont les plus importants du
projet :

```python
def _courbe(algo, mu):
    accs = []
    cfg = RunConfig(algo=algo, mu=mu, alpha=0.5, n_clients=4,
                    rounds=5, local_epochs=1, seed=0)
    run_federated("t", cfg, lambda m: accs.append(m.global_acc))
    return accs


def test_mu_zero_reproduit_fedavg():
    """Une erreur dans le terme proximal produit des courbes plausibles mais
    fausses. On ne le verrait jamais à l'oeil."""
    a = _courbe(Algo.fedavg, 0.0)
    b = _courbe(Algo.fedprox, 0.0)
    for x, y in zip(a, b):
        assert abs(x - y) < 1e-9


def test_mu_positif_change_le_resultat():
    """Attrape le cas où global_params serait mal passé : le terme s'annulerait
    silencieusement et FedProx redeviendrait FedAvg."""
    a = _courbe(Algo.fedavg, 0.0)
    b = _courbe(Algo.fedprox, 1.0)
    assert any(abs(x - y) > 1e-6 for x, y in zip(a, b))
```

> 🛑 **Point d'arrêt.** Si `μ=0` ne reproduit pas FedAvg **exactement**, le
> terme proximal est faux et tous les résultats FedProx sont à jeter.

**Balayage de μ :**

```bash
for mu in 0.001 0.01 0.1 1.0; do
  python scripts/check_federated.py --algo fedprox --mu $mu --alpha 0.1 --rounds 60
done
```

Attendu : un **μ optimal**, typiquement entre 0,01 et 0,1. Trop petit →
indiscernable de FedAvg. Trop grand → le modèle local est figé sur le global.

> ⚠️ **Si FedProx ≈ FedAvg partout**, ne concluez pas trop vite : il manque un
> régime, pas forcément un bug (les deux tests ci-dessus l'auraient attrapé).
> Le papier de Li et al. traite l'hétérogénéité **systèmes** autant que celle
> des données. Activez `systems_heterogeneity=True` : chaque client tire son
> nombre d'époques dans `[1, E]`. C'est le régime pour lequel le terme proximal
> a été conçu.

---

### ML-5 · Portage sur Flower
`ml/flower`

```bash
pip install "flwr[simulation]==1.33.0"
```

> ⚠️ **Ray est fragile sous Windows** et `flwr[simulation]` en dépend. Faites
> tourner la simulation **dans le conteneur Docker** (Linux) ou sur Colab.
>
> ⚠️ **Épinglez la version.** L'API a beaucoup changé entre 1.0 et 1.33 : les
> tutoriels trouvés en ligne ne correspondent pas.

**Ce que Flower ne fait PAS pour vous.** Sa stratégie `FedProx` se contente de
transmettre `proximal_mu` aux clients — vérifié dans sa source :

```python
# flwr/server/strategy/fedprox.py
return [(client, FitIns(fit_ins.parameters,
                        {**fit_ins.config, "proximal_mu": self.proximal_mu}))
        for client, fit_ins in client_config_pairs]
```

**C'est le `fit()` du client qui doit contenir le terme proximal.** Vous
réutilisez exactement `train_local()`.

**`fl_core/flower_runner.py`** — les points structurants :

```python
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from flwr.simulation import run_simulation


class FlowerClient(NumPyClient):
    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        mu = float(config.get("proximal_mu", 0.0))    # absent avec FedAvg -> 0
        global_params = [p.detach().clone() for p in self.model.parameters()]
        train_local(self.model, self.loader, epochs, lr, mu=mu,
                    global_params=global_params)
        return get_weights(self.model), self.n_samples, {}


def client_fn(context: Context):
    cid = int(context.node_config["partition-id"])    # relie le client à son shard
    ...


def evaluate_fn(server_round, parameters, config):
    """LE SEUL endroit où l'on journalise.

    Les clients tournent dans des processus Ray séparés : journaliser depuis
    eux produirait des écritures concurrentes ou des runs parasites.
    """
    ...
    on_round(RoundMetric(...))
    return loss, {"accuracy": acc}


strategy = FedProx(**common, proximal_mu=cfg.mu) if fedprox else FedAvg(**common)
run_simulation(server_app=ServerApp(server_fn=server_fn),
               client_app=ClientApp(client_fn=client_fn),
               num_supernodes=cfg.n_clients)
```

Mettre `fraction_evaluate=0.0` : l'évaluation distribuée est redondante avec la
nôtre et plus lente.

Enfin, basculer `get_runner()` dans `fl_core/runner.py` vers `FlowerRunner`.

> ✅ **Critère de fin :** à seed et configuration égales, les courbes Flower et
> manuelle se superposent. **Gardez les deux implémentations** — la manuelle
> devient un test de non-régression, et une validation croisée à citer au
> rapport.

---

### ML-6 · Rigueur expérimentale
`ml/metriques-seeds`

**`fl_core/metrics.py`**

```python
def final_accuracy(accs, window=5):
    """Moyenne des derniers rounds : lisse le bruit de fin de courbe."""

def rounds_to_target(accs, target):
    """Premier round atteignant la cible. None si jamais atteinte."""

def compare_arms(accs_a, accs_b):
    """Comparaison APPARIÉE sur les seeds : moyenne et écart-type de l'écart."""
```

**Trois seeds, pas un.** Le livrable exige de dire à partir de quel α FedProx
gagne *significativement*. Le mot est dans l'énoncé. Utilisez des **seeds
appariés** : même seed pour FedAvg et FedProx = même partition = comparaison à
données égales.

**Le budget de calcul.** `wall_time_s` est déjà journalisé. Multipliez le temps
par round par la taille de la grille : c'est cette mesure, pas une estimation,
qui décide si la grille finale tourne en local ou sur Colab.

---

### ML-7 · La grille finale
`ml/resultats-finaux`

**`experiments/grid.yaml`**

```yaml
dataset: mnist
n_clients: 10
rounds: 100
local_epochs: 2
lr: 0.01
batch_size: 64
target_acc: 0.90
seeds: [0, 1, 2]
alphas: [0.05, 0.1, 0.5, 1.0, 100.0]
arms:
  - {algo: fedavg,  mu: 0.0}
  - {algo: fedprox, mu: 0.01}
  - {algo: local,   mu: 0.0}
baselines:
  - {algo: centralized, mu: 0.0}
```

45 runs + 3 centralisés.

**`experiments/run_grid.py` doit être reprenable** : il interroge la base et
saute les runs déjà terminés. Sans ça, une déconnexion Colab à la 40ᵉ heure
fait tout recommencer.

**Colab** — le notebook ne contient **aucun code scientifique** :

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/justinadjassem/fl-noniid.git
%cd fl-noniid
!pip install -q -r requirements.txt && pip install -e .

import os
os.environ["MLFLOW_URI"] = "sqlite:////content/drive/MyDrive/fl-results/mlflow.db"
!python experiments/run_grid.py --config experiments/grid.yaml
```

> ⚠️ Colab déconnecte à ~90 min d'inactivité, 12 h maximum. Écrire dans Drive
> **à chaque round**, jamais seulement à la fin.
>
> ⚠️ **Colab n'est pas un environnement de développement**, c'est un four. On y
> exécute du code déjà juste, mis au point en local sur des runs minuscules.

**Rapatrier :**

```bash
cp ~/Downloads/mlflow.db mlruns/mlflow.db
git add -f mlruns/mlflow.db     # -f : ignoré pendant tout le développement
git commit -m "results: final ablation grid"
```

---

### BACK · Les tâches backend

| # | Branche | Contenu | Critère de fin |
|---|---|---|---|
| 1 | `api/service-mlflow` | `mlflow/Dockerfile`, service compose, `fl_core/tracking.py` | 3 services démarrent, UI MLflow sur `:5000` |
| 2 | `api/store-vers-mlflow` | remplacer le stockage mémoire par `search_runs()` | le dashboard marche **sans une ligne modifiée** |
| 3 | `api/routes-ingestion` | `POST /runs/external`, `/metrics`, `/complete` + jeton | un script extérieur peut verser des métriques |
| 4 | `api/streaming-sse` | flux SSE, worker séparé | la courbe se dessine round par round |
| 5 | `api/predict` | `POST /predict` : une image → 4 modèles | le client local se trompe, le global réussit |
| 6 | `api/docker-final` | vérification sur machine vierge | clone + une commande = ça marche |

**MLflow remplace le stockage mémoire, il ne s'y ajoute pas.** Deux sources de
vérité divergeraient. Ce qui reste à nous : `final_acc` et `rounds_to_target`,
qui sont de l'algorithme et vivent dans `fl_core/metrics.py`.

Le puits MLflow suit le même patron que le reste :

```python
def on_round(m: RoundMetric) -> None:
    mlflow.log_metrics({"global_acc": m.global_acc, ...}, step=m.round)
```

`step=m.round` est le point clé : MLflow est conçu pour des métriques indexées
par pas, et le round de communication en est un.

**BACK-5 mérite un mot** : `/predict` envoie la même image aux quatre modèles.
Le client local se trompe là où le global réussit — **c'est le client drift
rendu visible en direct**, et le meilleur moment de la soutenance.

---

### FRONT · Les tâches frontend

| # | Branche | Contenu | Critère de fin |
|---|---|---|---|
| 1 | `app/onglet-partition` | heatmap clients × classes | l'onglet existe, pas de régression |
| 2 | `app/bandes-incertitude` | moyenne ± σ sur les seeds, en aire | plusieurs seeds = une courbe + sa dispersion |
| 3 | `app/filtres` | filtrer par α, algorithme, dataset | lisible avec 60 runs en base |
| 4 | `app/tableau-croise` | algo × α, export | copiable tel quel dans le rapport |
| 5 | `app/demo-predict` | upload image → 4 verdicts | dépend de BACK-5 |

**FRONT n'attend personne** : le moteur factice produit des données conformes
au contrat. En attendant la vraie heatmap, générer une matrice 10×10 aléatoire —
à la fusion de ML-1 on remplace la **source de données**, pas l'affichage.

**Règle de couleur.** Les séries encodent une donnée :

```python
SERIES    = {"fedavg": "#2a78d6", "fedprox": "#eb6834"}
REFERENCE = "#898781"      # gris pointillé : centralisé et local pur
```

Validées pour la lisibilité daltonienne, et pour leur distance à l'accent
violet de l'interface. Le violet du thème ne rentre **jamais** dans un
graphique. Et un filtre qui change le nombre de séries ne doit pas repeindre
les survivantes.

---

## 7. Les pièges connus

Chacun a déjà coûté du temps à quelqu'un.

| # | Piège | Parade |
|---|---|---|
| 1 | **BatchNorm** fausse tout en fédéré non-IID | GroupNorm — déjà en place, protégé par un test |
| 2 | **`AdaptiveAvgPool((1,1))`** : 0,39 au lieu de 0,95, sans changer aucune forme | pooling en 4×4 — protégé par un test |
| 3 | **`global_params` rafraîchi** entre les batches : le terme proximal s'annule en silence | capturer une seule fois par round |
| 4 | **Un seul seed** : un écart de 1,5 point ne prouve rien | 3 seeds appariés minimum |
| 5 | **Clients vides** aux petits α | boucle de retirage — déjà en place |
| 6 | **Ray sous Windows** | simulation dans Docker ou Colab |
| 7 | **Développer dans Colab** | mise au point en local, Colab exécute |
| 8 | **`ModuleNotFoundError: contracts`** | `pip install -e .` |
| 9 | **Port 8501 occupé** | un `streamlit run` local tourne encore |
| 10 | **PowerShell écrit en UTF-16** | jamais de `>` pour créer un fichier texte |

---

## 8. Les livrables

| Livrable | Détail |
|---|---|
| **Rapport PDF** | 5-20 pages · schéma d'architecture **obligatoire** · répartition du travail détaillée · envoi à `tchaye59@gmail.com` |
| **Code GitHub** | dépôt propre · README installation + exécution · `mlruns/mlflow.db` commité |
| **Application** | web, fonctionnelle : tester, visualiser, interagir |
| **Présentation** | 15-20 min · **démonstration live** · architecture, choix, résultats |

**Éliminatoire :** plagiat = 0 · copie Kaggle sans adaptation = 0 · **modèle
seul sans application = REFUSÉ** · répartition non documentée = pénalité.

### Le livrable central

> « À partir de quel niveau d'hétérogénéité FedProx surpasse-t-il
> **significativement** FedAvg ? »

Ne répondez pas « à partir de α = 0,1 » sans plus. Répondez avec l'écart moyen
sur seeds appariés et son écart-type, et dites à partir d'où l'écart dépasse la
variabilité inter-seeds. **Si l'écart n'est jamais significatif sans
hétérogénéité systèmes, dites-le** — c'est un résultat honnête, et le rapporter
vaut mieux que le maquiller.

### Les cinq choix méthodologiques à défendre

1. **GroupNorm plutôt que BatchNorm** — moyenner des statistiques de batch
   entre clients non-IID produit des valeurs qui ne correspondent à aucun client
2. **Borne basse évaluée sur le test global** — seule mesure comparable à FedAvg
3. **`final_acc` = moyenne des 5 derniers rounds** — lisse le bruit
4. **Seeds appariés** — même seed = même partition = comparaison à données égales
5. **Validation croisée maison ↔ Flower** — deux implémentations qui concordent

---

## 9. Récapitulatif des points d'arrêt

| Étape | Le test | Si ça échoue |
|---|---|---|
| ML-2 | **~99 % MNIST centralisé** | le modèle est cassé, ne pas écrire de code fédéré |
| ML-3 | **98-99 % en quasi-IID** | l'agrégation est cassée, pas l'hétérogénéité |
| ML-4 | **μ=0 ≡ FedAvg exactement** | le terme proximal est faux, résultats à jeter |
| BACK-6 | clone + une commande | « s'il faut 2 h de configuration, c'est raté » |

Ces tests coûtent quelques minutes chacun et protègent l'intégralité du projet.
Le troisième est le plus dangereux : les courbes restent plausibles.
