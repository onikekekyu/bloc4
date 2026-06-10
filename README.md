# Fish Classification — MLOps Project

Classification d'images de poissons en 5 espèces avec un pipeline MLOps complet.

- **Modèle :** ResNet18 (transfer learning) — 84% validation accuracy
- **Dataset :** 1306 images (Catfish, Goldfish, Mudfish, Mullet, Snakehead)
- **Stack :** FastAPI · React/Vite · MinIO · MySQL · MLflow · Prometheus · Grafana · Docker

---

## Démarrage rapide

```bash
docker compose up -d
```

L'API attend que MinIO et MySQL soient healthy avant de démarrer. Tout est prêt en ~30 secondes.

| Service | URL | Identifiants |
|---|---|---|
| Frontend React | http://localhost:3000 | — |
| API FastAPI | http://localhost:8000 | — |
| Grafana | http://localhost:3002 | `admin` / `admin` |
| Prometheus | http://localhost:9090 | — |
| MinIO console | http://localhost:9001 | `admin-user` / `admin-password` |
| phpMyAdmin | http://localhost:8080 | `root` / `root` |
| MLflow | http://localhost:5001 | — |
| cAdvisor | http://localhost:8081 | — |

---

## Structure

```
.
├── api/                        # API FastAPI (serving)
│   ├── main.py                 # Endpoints : /predict, /drift, /drift/report, /metrics
│   ├── model.py                # Chargement ResNet18 depuis MinIO
│   ├── db.py                   # Connexion MySQL, log des prédictions
│   ├── drift.py                # Rapport Evidently (détection de drift)
│   └── requirements.txt
├── src/                        # Scripts ML
│   ├── extraction_creation_sql.py  # Extraction métadonnées MinIO → MySQL
│   ├── train_model.py              # Entraînement ResNet18 + MLflow tracking
│   ├── predict.py                  # Test sur le jeu de test
│   └── upload_model_to_minio.py
├── retrain/
│   └── retrain.py              # Réentraînement automatisé (drift-triggered)
├── tests/
│   └── test_predict.py         # Tests d'intégration API
├── frontend/                   # React + Vite
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/provisioning/   # Dashboard + datasource auto-provisionnés
├── k8s/                        # Manifests Kubernetes (désactivés)
├── .github/workflows/
│   ├── ci-cd.yml               # Build, test, push images Docker → GHCR
│   └── retrain.yml             # Réentraînement manuel ou hebdomadaire (cron)
├── Dockerfile                  # Image Python pour les scripts ML
└── docker-compose.yml          # Orchestration complète
```

---

## Pipeline ML

```
MinIO (dataset-fish)
    ↓
extraction_creation_sql.py  →  MySQL (table fish_data, champ split train/test)
    ↓
train_model.py  →  ResNet18 fine-tuning (20 epochs, Adam, LR=0.001)
                →  MLflow tracking (params + métriques par epoch)
                →  Upload modèle versionné dans MinIO (model_v1_{timestamp}.pt)
    ↓
API FastAPI  →  /predict (POST image → {prediction, confidence})
            →  /drift   (GET → rapport Evidently JSON)
            →  /drift/report (GET → rapport HTML complet)
            →  /metrics (GET → métriques Prometheus)
```

### Paramètres d'entraînement

| Paramètre | Valeur |
|---|---|
| Architecture | ResNet18 (IMAGENET1K_V1) |
| Epochs | 20 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Split train/val | 80/20 |
| Meilleure val accuracy | 84.21% |

---

## Réentraînement automatisé

Le workflow `.github/workflows/retrain.yml` se déclenche :
- **Manuellement** via GitHub Actions → workflow_dispatch
- **Automatiquement** chaque lundi à 2h (cron)

Le script `retrain/retrain.py` vérifie d'abord si un drift est détecté (confiance moyenne < 70% sur les 7 derniers jours) avant de relancer le pipeline. Il peut être forcé avec `--force`.

```bash
# Lancer manuellement le réentraînement
python retrain/retrain.py --force
```

---

## Monitoring

**Prometheus** scrape l'API toutes les 15 secondes sur `/metrics`.

Métriques exposées :
- `fish_predictions_total` — compteur par classe prédite
- `fish_prediction_duration_seconds` — histogramme des temps de réponse
- `fish_prediction_confidence` — dernière confiance par classe
- `fish_prediction_errors_total` — compteur d'erreurs

**Grafana** (http://localhost:3002) charge automatiquement le dashboard "Fish Classifier API Metrics" avec 5 panneaux :
1. Taux de prédictions par classe (req/s)
2. Durée de prédiction p95
3. Confiance par classe (dernière valeur)
4. Erreurs totales
5. Répartition des espèces prédites (camembert)

**Evidently** — détection de drift sur les prédictions récentes vs la distribution de référence du training :
- `GET /drift` → résumé JSON (drift détecté ou non)
- `GET /drift/report` → rapport HTML interactif complet

---

## CI/CD

Le workflow `.github/workflows/ci-cd.yml` se déclenche sur chaque push sur `main` :

1. Tests Python (pytest) sur l'API
2. Build et vérification du frontend (npm build)
3. Build et push des images Docker vers GitHub Container Registry (GHCR)

Images publiées :
- `ghcr.io/<owner>/fishy-api:latest`
- `ghcr.io/<owner>/fishy-frontend:latest`

---

## Pipeline complet (réentraînement depuis zéro)

```bash
# 1. Infrastructure
docker compose up -d minio mysql mlflow

# 2. Extraction + entraînement (attend que les services soient healthy)
docker compose up --build extraction training

# 3. Serving + monitoring
docker compose up -d fish_api frontend prometheus grafana cadvisor
```

## Anti-data leakage

- Champ `split` (train/test) dans la table MySQL `fish_data`
- Entraînement sur `WHERE split = 'train'` uniquement
- Validation 80/20 sur les données d'entraînement
- Test sur `WHERE split = 'test'` uniquement
- Meilleur modèle sélectionné sur la validation accuracy

## Nettoyage

```bash
# Arrêter tout
docker compose down

# Arrêter et supprimer les volumes (repart de zéro)
docker compose down -v
```
