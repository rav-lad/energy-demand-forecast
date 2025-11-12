# 🐳 Docker Guide - Energy Trading Research

## Vue d'ensemble

Le projet est entièrement containerisé avec Docker pour garantir :
- ✅ Environnement reproductible
- ✅ Isolation des dépendances
- ✅ Déploiement facile
- ✅ Support multi-services

---

## 📦 Architecture Docker

### Services disponibles

```yaml
Services:
  app         → Application principale (CLI)
  jupyter     → Jupyter Lab pour recherche
  train       → Service d'entraînement de modèles
  benchmark   → Service de benchmark
  data-collector → Collecte de données
  backtest    → Backtesting de stratégies
```

### Images

- **Base** : `python:3.10-slim`
- **Production** : `energy-trading:latest`
- **Development** : `energy-trading:dev` (avec Jupyter)

---

## 🚀 Quick Start

### 1. Build l'image

```bash
# Image standard (CPU)
docker-compose build app

# Ou build directement avec Docker
docker build -t energy-trading:latest .
```

### 2. Configuration

```bash
# Copier le template d'environnement
cp .env.example .env

# Éditer et ajouter votre clé ENTSO-E
nano .env
```

### 3. Lancer un service

```bash
# Application interactive
docker-compose up app

# Jupyter Lab (port 8888)
docker-compose up jupyter

# Training
docker-compose run train

# Benchmark
docker-compose run benchmark
```

---

## 💻 Utilisation

### Mode interactif (CLI)

```bash
# Démarrer container interactif
docker-compose run --rm app bash

# Une fois dans le container
python run_backtest_example.py
python scripts/train_pipeline.py --model xgboost --frequency daily
python scripts/benchmark_models.py --frequency daily
```

### Jupyter Lab

```bash
# Démarrer Jupyter
docker-compose up jupyter

# Ouvrir dans le navigateur
http://localhost:8888

# Token par défaut : aucun (désactivé pour dev)
```

### Entraînement de modèles

```bash
# Entraîner tous les modèles
docker-compose run train python scripts/train_pipeline.py --model all --frequency daily

# Entraîner un modèle spécifique
docker-compose run train python scripts/train_pipeline.py --model xgboost --frequency daily

# Avec GPU (si configuré)
docker-compose run --gpus all train python scripts/train_pipeline.py --model tft --frequency daily --gpus 1
```

### Benchmark

```bash
# Lancer le benchmark
docker-compose run benchmark

# Ou avec options personnalisées
docker-compose run benchmark python scripts/benchmark_models.py --frequency daily --models xgboost lightgbm
```

### Collecte de données

```bash
# Démarrer service de collecte
docker-compose run data-collector bash

# Collecter prix de marché
python data_recuperation/data_market_prices.py \
    --start_date 2020-01-01 \
    --end_date 2024-12-31 \
    --countries FR DE ES

# Collecter données fondamentales
python data_recuperation/data_fundamentals.py \
    --country FR \
    --data_type all
```

### Backtesting

```bash
# Lancer backtest
docker-compose run backtest

# Avec configuration personnalisée
docker-compose run backtest python run_backtest_example.py
```

---

## 📁 Volumes

### Volumes montés automatiquement

```yaml
Volumes:
  ./data:/app/data           # Données persistantes
  ./models:/app/models       # Modèles entraînés
  ./outputs:/app/outputs     # Résultats
  ./research:/app/research   # Notebooks de recherche
```

### Gestion des données

Les données sont **persistées sur l'hôte** :
- ✅ Pas besoin de recréer les données à chaque build
- ✅ Facile à backup
- ✅ Partagé entre tous les containers

```bash
# Les données restent après suppression du container
docker-compose down
# → data/, models/, outputs/ restent intacts
```

---

## 🔧 Configuration avancée

### Variables d'environnement

Définies dans `.env` :

```bash
# API Keys
ENTSOE_API_KEY=your_key_here

# Application
DEBUG=False
LOG_LEVEL=INFO
PYTHONPATH=/app

# Jupyter (optionnel)
JUPYTER_ENABLE_LAB=yes
```

### Support GPU

Pour utiliser GPU avec TFT :

1. **Installer NVIDIA Container Toolkit**

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. **Décommenter dans `docker-compose.yml`**

```yaml
train:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

3. **Build avec support GPU**

```bash
docker build --build-arg USE_GPU=true -t energy-trading:gpu .
```

4. **Utiliser**

```bash
docker-compose run --gpus all train \
    python scripts/train_pipeline.py --model tft --gpus 1
```

### Build multi-stage

Le Dockerfile utilise plusieurs stages :

```dockerfile
base → dependencies → python-deps → application → final
                                                 ↓
                                          development (avec Jupyter)
                                                 ↓
                                          production (minimal)
```

**Build development** :
```bash
docker build --target development -t energy-trading:dev .
```

**Build production** :
```bash
docker build --target production -t energy-trading:prod .
```

---

## 🔍 Debugging

### Inspecter un container en cours

```bash
# Lister containers actifs
docker ps

# Entrer dans un container
docker exec -it energy-trading-app bash

# Voir les logs
docker logs energy-trading-app

# Suivre les logs en temps réel
docker logs -f energy-trading-jupyter
```

### Vérifier les volumes

```bash
# Lister les volumes
docker volume ls

# Inspecter un volume
docker volume inspect energy-demand-forecast_data-volume
```

### Résoudre problèmes courants

#### Problème : "Permission denied"

```bash
# Solution : Ajouter permissions au dossier
chmod -R 755 data/ models/ outputs/
```

#### Problème : "Port already in use"

```bash
# Changer le port dans docker-compose.yml
ports:
  - "8889:8888"  # Au lieu de 8888:8888
```

#### Problème : "Out of memory"

```bash
# Allouer plus de mémoire à Docker
# Docker Desktop → Settings → Resources → Memory → 8 GB
```

#### Problème : Build lent

```bash
# Utiliser cache de build
docker-compose build --parallel

# Nettoyer images inutilisées
docker system prune -a
```

---

## 📊 Workflow complet avec Docker

### 1. Setup initial

```bash
# Build
docker-compose build

# Configuration
cp .env.example .env
nano .env  # Ajouter ENTSOE_API_KEY
```

### 2. Collecte de données

```bash
docker-compose run data-collector bash

# Dans le container
python data_recuperation/data_market_prices.py --start_date 2020-01-01 --end_date 2024-12-31 --countries FR
python data_recuperation/data_fundamentals.py --country FR --data_type all
```

### 3. Preprocessing

```bash
docker-compose run app bash

# Dans le container
python data_processing/transformation.py
python data_processing/split_train_test.py
```

### 4. Entraînement

```bash
docker-compose run train \
    python scripts/train_pipeline.py --model all --frequency daily
```

### 5. Benchmark

```bash
docker-compose run benchmark
```

### 6. Backtesting

```bash
docker-compose run backtest
```

### 7. Analyse (Jupyter)

```bash
docker-compose up jupyter
# → Ouvrir http://localhost:8888
```

---

## 🌐 Production Deployment

### Option 1 : Docker Compose (simple)

```bash
# Sur le serveur
git clone https://github.com/yourusername/energy-demand-forecast.git
cd energy-demand-forecast

# Configuration
cp .env.example .env
nano .env

# Build production
docker-compose build

# Lancer services
docker-compose up -d app
```

### Option 2 : Docker Swarm (scalable)

```bash
# Initialiser Swarm
docker swarm init

# Déployer stack
docker stack deploy -c docker-compose.yml energy-trading

# Scaler services
docker service scale energy-trading_train=3
```

### Option 3 : Kubernetes (enterprise)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: energy-trading
spec:
  replicas: 3
  selector:
    matchLabels:
      app: energy-trading
  template:
    metadata:
      labels:
        app: energy-trading
    spec:
      containers:
      - name: app
        image: energy-trading:latest
        ports:
        - containerPort: 8050
        env:
        - name: ENTSOE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: entsoe
```

```bash
kubectl apply -f k8s/deployment.yaml
```

---

## 📈 Performance

### Optimisations

#### 1. Build cache

```bash
# Utiliser BuildKit pour builds parallèles
DOCKER_BUILDKIT=1 docker build -t energy-trading:latest .
```

#### 2. Multi-stage pour images plus petites

```bash
# Production image (sans dev tools)
docker build --target production -t energy-trading:prod .

# Taille:
# - development: ~2.5 GB
# - production:  ~1.2 GB
```

#### 3. Volumes nommés pour performance

```yaml
volumes:
  data-volume:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/fast-ssd/data
```

---

## 🔒 Sécurité

### Best practices

#### 1. Ne pas commit les secrets

```bash
# .gitignore déjà configuré
.env
.env.local
```

#### 2. Utiliser Docker secrets

```bash
# Créer secret
echo "your_api_key" | docker secret create entsoe_key -

# Dans docker-compose.yml
secrets:
  entsoe_key:
    external: true
```

#### 3. Scan vulnérabilités

```bash
# Avec Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image energy-trading:latest
```

#### 4. Utiliser utilisateur non-root

```dockerfile
# Déjà configuré dans production stage
USER appuser
```

---

## 🧹 Maintenance

### Nettoyage

```bash
# Arrêter tous les containers
docker-compose down

# Supprimer images inutilisées
docker image prune -a

# Supprimer volumes inutilisés (ATTENTION: perte de données)
docker volume prune

# Nettoyage complet
docker system prune -a --volumes
```

### Mise à jour

```bash
# Pull dernières images
docker-compose pull

# Rebuild avec cache
docker-compose build --pull

# Restart services
docker-compose up -d
```

### Backup

```bash
# Backup volumes
docker run --rm -v energy-demand-forecast_data-volume:/data \
    -v $(pwd)/backups:/backup \
    ubuntu tar czf /backup/data-backup-$(date +%Y%m%d).tar.gz /data

# Restore
docker run --rm -v energy-demand-forecast_data-volume:/data \
    -v $(pwd)/backups:/backup \
    ubuntu tar xzf /backup/data-backup-20240101.tar.gz -C /
```

---

## 📚 Ressources

### Documentation Docker
- Official Docs: https://docs.docker.com/
- Compose Docs: https://docs.docker.com/compose/
- Best Practices: https://docs.docker.com/develop/dev-best-practices/

### Alternatives
- **Podman**: Drop-in replacement for Docker (rootless)
- **Kubernetes**: Orchestration pour production
- **Singularity**: HPC environments

---

## 🆘 Support

### Problèmes communs

| Problème | Solution |
|----------|----------|
| Container ne démarre pas | Vérifier logs: `docker logs <container>` |
| Erreur API key | Vérifier `.env` existe et contient `ENTSOE_API_KEY` |
| Out of memory | Augmenter RAM Docker (Settings → Resources) |
| Permission denied | `chmod -R 755 data/ models/ outputs/` |
| Port déjà utilisé | Changer port dans `docker-compose.yml` |

### Logs

```bash
# Voir tous les logs
docker-compose logs

# Logs d'un service spécifique
docker-compose logs jupyter

# Suivre logs en temps réel
docker-compose logs -f train
```

---

## ✅ Checklist de déploiement

### Développement
- [ ] Build image: `docker-compose build`
- [ ] Copier `.env`: `cp .env.example .env`
- [ ] Ajouter API key dans `.env`
- [ ] Tester : `docker-compose run app python --version`
- [ ] Lancer Jupyter : `docker-compose up jupyter`

### Production
- [ ] Build production: `docker build --target production`
- [ ] Configurer secrets (pas `.env`)
- [ ] Setup monitoring (Prometheus, Grafana)
- [ ] Setup logging (ELK stack)
- [ ] Configurer backup automatiques
- [ ] Scanner vulnérabilités
- [ ] Load testing
- [ ] Documentation deployment

---

**Dernière mise à jour** : 2024-11-12

Pour questions ou problèmes, créer une issue sur GitHub.
