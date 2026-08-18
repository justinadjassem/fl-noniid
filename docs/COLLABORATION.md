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
| **ML-2** modèle et bornes | ✅ | **99,33 % en centralisé** — point d'arrêt franchi |

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

Borne haute mesurée — MNIST centralisé, 8 époques, lr 0,01, batch 64 :

| round | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| accuracy | 0,9834 | 0,9890 | 0,9898 | 0,9910 | **0,9933** |

L'accuracy oscille de ±0,2 point d'une époque à l'autre, **en centralisé, dans
le cas le plus favorable qui soit**. C'est votre bruit de fond de référence :
un écart de cet ordre entre FedAvg et FedProx ne prouvera rien. Ça justifie
concrètement `final_acc` sur la moyenne des 5 derniers rounds, et les 3 seeds.

### ⚠️ Le budget de calcul, à trancher en ML-6

**296 s par époque** sur les 60 000 images (2 371 s / 8 époques, CPU).

```
10 clients × 2 époques locales × 6 000 images = 120 000 images/round
120 000 / 60 000 × 296 s                      ≈ 590 s par round  (~10 min)
```

Un run de 100 rounds = 16 h. La grille de 45 runs = plusieurs semaines. **C'est
intenable en l'état**, et l'ajout du mode asynchrone double le nombre de runs.

Les leviers, à combiner :

| Levier | Gain |
|---|---|
| `batch_size` 64 → 128 | ×1,5 |
| sous-échantillonner MNIST à 20 000 images | ×3 |
| `participation` 1,0 → 0,5 | ×2 |
| `rounds` 100 → 60 | ×1,7 |
| Colab GPU | ×3-5 |

Deux ou trois de ces leviers suffisent à repasser sous la nuit. **La décision
se prend sur `wall_time_s` mesuré, pas sur une estimation** — c'est pour ça que
le champ est dans le contrat depuis le premier jour.

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

### ML-8 · Monitoring par client
`ml/monitoring-clients` — *dépend de ML-3*

Jusqu'ici on ne journalise que l'agrégat par round. Il faut descendre au client.

**Le contrat change.** Ajout dans `contracts/schemas.py` :

```python
class ClientMetric(BaseModel):
    """Une ligne par client et par round. C'est le grain fin du monitoring."""

    client_id: int
    n_samples: int              # taille du shard : le déséquilibre de quantité
    epochs_run: int             # < local_epochs si hétérogénéité systèmes
    local_acc: float            # sur le jeu de test GLOBAL, comparable
    local_loss: float
    drift: float                # ||w_k - w_global||_2  <-- voir ci-dessous
    wall_time_s: float          # qui ralentit le round


class RoundMetric(BaseModel):
    ...
    clients: list[ClientMetric] = []     # défaut vide : changement NON cassant
```

> Le défaut `[]` rend la modification rétrocompatible : le moteur factice, les
> bornes et le dashboard continuent de fonctionner sans une ligne modifiée.
> C'est la forme à privilégier pour toute évolution du contrat.

**La métrique qui compte : `drift`**

```python
drift = torch.sqrt(sum(((p - g) ** 2).sum()
                       for p, g in zip(local.parameters(), global_params))).item()
```

C'est **exactement la quantité que le terme proximal de FedProx pénalise**.
La journaliser par client et par round vous donne bien plus qu'une courbe
d'accuracy : la preuve directe que FedProx *contient la dérive*, et pas
seulement qu'il améliore le score. Attendu — à vérifier :

- à α grand, `drift` reste faible et homogène entre clients
- à α petit, `drift` explose et se disperse
- **à μ > 0, `drift` doit être visiblement plus faible qu'à μ = 0, à α égal**

Si ce dernier point n'est pas observé, votre terme proximal ne fait pas son
travail — c'est un test plus fin que la comparaison d'accuracy.

**Où journaliser.** Toujours **côté serveur**. Les clients Flower tournent dans
des processus Ray séparés : écrire depuis eux produirait des écritures
concurrentes. Mais `fit()` renvoie un dictionnaire de métriques :

```python
def fit(self, parameters, config):
    ...
    return get_weights(self.model), self.n_samples, {
        "client_id": self.cid, "local_acc": acc, "local_loss": loss,
        "epochs_run": epochs, "drift": drift, "wall_time_s": dt,
    }
```

Ces dictionnaires remontent au serveur via `fit_metrics_aggregation_fn`, et
c'est le serveur qui écrit dans MLflow :

```python
for c in metric.clients:
    mlflow.log_metrics({
        f"client_{c.client_id}/local_acc": c.local_acc,
        f"client_{c.client_id}/drift":     c.drift,
        f"client_{c.client_id}/epochs":    c.epochs_run,
    }, step=metric.round)
```

10 clients × 6 métriques × 100 rounds = 6 000 points par run. MLflow encaisse
sans difficulté.

> ✅ **Critère de fin :** dans l'UI MLflow, sélectionner un run et voir dix
> courbes `client_*/drift` superposées. À α=0.05 elles doivent diverger
> visiblement ; à α=100 rester groupées.

---

### ML-9 · Mode asynchrone et comparaison
`ml/async-server` — *dépend de ML-3 et ML-8*

**Pourquoi le serveur maison sert enfin à autre chose.** Flower en simulation
est **synchrone par construction** : la stratégie attend tous les clients
sélectionnés avant d'agréger. L'asynchrone ne s'y exprime pas naturellement.
Votre boucle manuelle de ML-3 en devient le support — elle a désormais deux
raisons d'exister : oracle de validation, et substrat de l'asynchrone.

**`fl_core/async_server.py`** — approche FedAsync (Xie et al. 2019) :

```
w_global <- (1 - a_t) * w_global + a_t * w_k
```

où `a_t = alpha_async * s(tau)`, avec `tau = round_courant - round_de_depart_du_client`
la **staleness** (ancienneté de la mise à jour), et `s` décroissante :

```python
def staleness_weight(tau: int, kind: str = "polynomial", a: float = 0.5) -> float:
    """Pondération décroissante avec l'ancienneté de la mise à jour."""
    if kind == "constant":
        return 1.0
    if kind == "polynomial":
        return (1 + tau) ** (-a)
    if kind == "hinge":
        return 1.0 if tau <= 4 else 1.0 / (a * (tau - 4) + 1)
    raise ValueError(kind)
```

**Simuler l'asynchronisme sans concurrence réelle.** Pas besoin de threads :
on donne à chaque client une *durée* de calcul (tirée d'une loi, ou
proportionnelle à la taille de son shard), on maintient une file d'événements
triée par date de fin, et on agrège **dès qu'un client termine** au lieu
d'attendre le round complet. Déterministe, reproductible, et ça capture
l'essentiel : la staleness.

```
file = [(date_fin_k, client_k, w_global_au_depart) pour chaque client]
tant que temps < budget :
    prendre l'événement le plus proche
    tau = round_courant - round_de_depart_de_ce_client
    w_global <- (1 - a_t) * w_global + a_t * w_k     avec a_t = alpha_async * s(tau)
    replanifier ce client avec une nouvelle durée
```

**Les runs à produire pour la comparaison :**

| Configuration | Pourquoi |
|---|---|
| sync FedAvg, α ∈ {0.1, 0.5} | référence |
| async FedAvg, α ∈ {0.1, 0.5} | l'effet du protocole |
| sync FedProx μ=0.01 | référence |
| async FedProx μ=0.01 | le terme proximal aide-t-il aussi en async ? |
| les quatre, avec `systems_heterogeneity=True` | le régime où l'async doit gagner |

À comparer **à temps de calcul égal**, pas à nombre de rounds égal — c'est tout
l'argument de l'asynchrone : plus de mises à jour par unité de temps.

> ✅ **Critère de fin :** une figure accuracy vs *temps écoulé* (pas vs round)
> montrant sync et async, avec et sans stragglers.

> ⚠️ **Le piège de la comparaison.** Comparer sync et async à nombre de rounds
> égal est trompeur et biaise en faveur du synchrone. L'axe des abscisses doit
> être le temps, ou le nombre de mises à jour du modèle global.

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
| **7** | `api/model-registry` | packaging MLflow + registry + alias | `models:/fl-noniid-fedavg@champion` résout |
| **8** | `api/model-serving` | conteneur `mlflow models serve` | `POST /invocations` répond une prédiction |

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

#### BACK-7 · Packaging et Model Registry

Journaliser le modèle avec **signature et exemple d'entrée** — sans eux,
MLflow ne sait pas valider ce qu'on lui envoie au moment de servir :

```python
import mlflow, numpy as np
from mlflow.models import infer_signature

example = np.random.rand(1, 1, 28, 28).astype("float32")
with torch.no_grad():
    signature = infer_signature(example, model(torch.tensor(example)).numpy())

mlflow.pytorch.log_model(
    pytorch_model=model,
    name="model",
    signature=signature,
    input_example=example,
    registered_model_name=f"fl-noniid-{cfg.algo.value}",
)
```

> ⚠️ **Les « stages » du registry (`Staging` / `Production`) sont dépréciés
> depuis MLflow 2.9** — `transition_model_version_stage` porte un
> `@deprecated`. Le remplacement officiel, ce sont les **alias**. Même rôle :
> désigner quelle version est servie, sans coder un numéro en dur.

```python
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias("fl-noniid-fedavg", "champion", version="3")
client.set_registered_model_alias("fl-noniid-fedavg", "challenger", version="4")

# à citer au rapport : l'URI de service ne change jamais, seule la cible bouge
uri = "models:/fl-noniid-fedavg@champion"
```

Taguer les versions avec les métriques qui ont motivé la promotion :

```python
client.set_model_version_tag("fl-noniid-fedavg", "3", "final_acc", "0.9412")
client.set_model_version_tag("fl-noniid-fedavg", "3", "alpha", "0.1")
```

> ✅ **Critère de fin :** `mlflow.pyfunc.load_model("models:/fl-noniid-fedavg@champion")`
> charge le modèle et prédit sur une image.

#### BACK-8 · Serving via `mlflow models serve`

Un service par modèle servi. **Réutilisez l'image de l'API** : elle contient
déjà torch et mlflow, donc le modèle se désérialise. Pas de nouvelle image.

```yaml
  serve-fedavg:
    build: {context: ., dockerfile: api/Dockerfile}
    command: >
      mlflow models serve
      --model-uri models:/fl-noniid-fedavg@champion
      --host 0.0.0.0 --port 5001
      --env-manager local
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports: ["5001:5001"]
    volumes: ["./mlruns:/mlflow"]      # les artefacts doivent être lisibles
    depends_on: [mlflow]
```

Trois pièges, tous coûteux :

| Piège | Parade |
|---|---|
| MLflow recrée un environnement conda par modèle au démarrage | `--env-manager local` : l'image a déjà torch |
| `models:/…` ne résout pas | `MLFLOW_TRACKING_URI` doit pointer le serveur de suivi |
| Le serveur démarre mais ne trouve pas les poids | monter le volume des artefacts |

**Appeler l'endpoint :**

```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[[[0.0, 0.1, ...]]]]}'      # forme (1, 1, 28, 28)
```

> ✅ **Critère de fin :** `POST /invocations` renvoie dix logits pour une image
> MNIST. À montrer en soutenance — c'est l'endpoint MLflow natif, pas un
> wrapper maison.

**Ce que devient `/predict`.** L'API FastAPI garde son rôle de démonstration
comparée : elle appelle les endpoints `/invocations` des quatre services et
renvoie les quatre verdicts côte à côte. MLflow sert les modèles, FastAPI
orchestre la comparaison.

---

### FRONT · Les tâches frontend

| # | Branche | Contenu | Critère de fin |
|---|---|---|---|
| 1 | `app/onglet-partition` | heatmap clients × classes | l'onglet existe, pas de régression |
| 2 | `app/bandes-incertitude` | moyenne ± σ sur les seeds, en aire | plusieurs seeds = une courbe + sa dispersion |
| 3 | `app/filtres` | filtrer par α, algorithme, dataset | lisible avec 60 runs en base |
| 4 | `app/tableau-croise` | algo × α, export | copiable tel quel dans le rapport |
| 5 | `app/demo-predict` | upload image → 4 verdicts | dépend de BACK-5 et BACK-8 |
| **6** | `app/vue-par-client` | onglet Clients : une courbe par client | dépend de ML-8 |

**FRONT-6** est l'onglet le plus riche du dashboard. Pour un run donné, dix
courbes de `drift` et dix d'`local_acc`, plus le nombre d'époques réellement
effectuées par client et par round (qui révèle les stragglers).

Règle de couleur ici : dix clients dépassent le nombre de teintes catégorielles
distinguables. **N'inventez pas dix couleurs.** Une seule teinte, opacité
faible pour l'ensemble, et **mise en évidence du client survolé** — le message
est la dispersion du faisceau, pas l'identité de chaque courbe.

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

### Livrable d'analyse · Synchrone vs asynchrone

Section du rapport et des slides, **adossée à vos propres mesures** — c'est ce
qui la distinguera d'un copier-coller de cours. Les axes :

| | **Synchrone** | **Asynchrone** |
|---|---|---|
| **Protocole** | le serveur attend les K clients sélectionnés, puis agrège | le serveur agrège dès qu'un client rend son travail |
| **Vitesse d'un round** | celle du **plus lent** | indépendante des lents |
| **Stragglers** | bloquants : un client à 0,25 CPU impose son rythme à tous | absorbés |
| **Pannes / déconnexions** | round perdu ou client exclu | sans effet, les autres continuent |
| **Fraîcheur des mises à jour** | toutes calculées depuis le **même** modèle global | **staleness** : un client lent rend une mise à jour calculée depuis un modèle périmé |
| **Convergence** | garanties théoriques établies (FedAvg, FedProx) | plus fragile, nécessite une pondération par ancienneté |
| **Reproductibilité** | déterministe à seed égale | dépend de l'ordre d'arrivée — difficile à reproduire exactement |
| **Débogage** | simple : un round, un état | difficile : l'état global change pendant qu'un client calcule |
| **Débit** | faible si les clients sont hétérogènes | élevé |
| **Coût de communication par round** | K × 2 × taille du modèle, prévisible | continu, plus difficile à budgéter |

**L'argument central, à formuler avec vos chiffres :** en synchrone, le round
coûte le temps du client le plus lent. Avec `systems_heterogeneity`, mesurez
`max(wall_time_s)` et `mean(wall_time_s)` par round — le rapport des deux **est**
le coût du protocole synchrone, chiffré sur votre propre expérience.

**Le lien avec FedProx.** Le terme proximal rend les mises à jour *partielles*
agrégeables sans casser le modèle. C'est une réponse **au sein du protocole
synchrone** au même problème que l'asynchrone traite en changeant de protocole.
Les deux approches attaquent l'hétérogénéité systèmes par des voies opposées :
à discuter, c'est le genre de mise en perspective qui distingue un rapport.

**Ce qui reste vrai dans les deux cas :** l'hétérogénéité des *données* (votre
α) n'est pas réglée par le choix du protocole. L'asynchrone traite les
stragglers, pas le client drift.

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
| 11 | **Stages MLflow dépréciés** depuis 2.9 | utiliser les **alias** (`@champion`) |
| 12 | **`mlflow models serve` recrée un env conda** au démarrage | `--env-manager local` |
| 13 | **Journaliser depuis les clients Ray** : écritures concurrentes | renvoyer les métriques depuis `fit()`, journaliser côté serveur |
| 14 | **Comparer sync et async à nombre de rounds égal** : biaise en faveur du synchrone | axe des abscisses = temps écoulé |

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
| ML-2 | **~99 % MNIST centralisé** ✅ *(99,33 %)* | le modèle est cassé, ne pas écrire de code fédéré |
| ML-3 | **98-99 % en quasi-IID** | l'agrégation est cassée, pas l'hétérogénéité |
| ML-4 | **μ=0 ≡ FedAvg exactement** | le terme proximal est faux, résultats à jeter |
| ML-8 | **`drift` plus faible à μ>0 qu'à μ=0**, à α égal | le terme proximal ne contient rien |
| BACK-8 | **`POST /invocations` répond** | le packaging ou le registry est cassé |
| BACK-6 | clone + une commande | « s'il faut 2 h de configuration, c'est raté » |

Ces tests coûtent quelques minutes chacun et protègent l'intégralité du projet.
Le troisième est le plus dangereux : les courbes restent plausibles.
