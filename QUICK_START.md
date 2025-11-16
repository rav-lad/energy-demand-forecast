# 🚀 QUICK START - Données Réelles

**Le projet est 100% prêt !** Vous n'avez besoin que d'**une seule clé API** (ENTSO-E, gratuite).

---

## ✅ Ce qui fonctionne SANS clé API

Ces sources de données fonctionnent **immédiatement**, sans aucune configuration :

### 1. Météo (Open-Meteo) - 0 min setup ✨
```bash
# Données historiques météo (toutes régions FR)
python data_collection/pipeline.py weather-historical --frequency daily

# Prévisions météo
python data_collection/pipeline.py weather-forecast --frequency hourly
```

### 2. Consommation Énergie (ODRE) - 0 min setup ✨
```bash
# Consommation électricité + gaz (toutes régions FR)
python data_collection/odre_collector.py \
  --start_date 2022-01-01 \
  --end_date 2024-12-31 \
  --validate
```

**Ces 2 sources représentent ~40% des données du projet et fonctionnent MAINTENANT !**

---

## 🔑 Ce qui nécessite une clé API

Une seule clé pour 2 sources de données critiques :

### 3. Prix Électricité (ENTSO-E) - 1 clé API requise 🔑

**Obtenir la clé (5 min, activation 24-48h) :**

1. **S'inscrire :** https://transparency.entsoe.eu/
2. **Générer clé :** My Account → Generate API Key
3. **Attendre activation :** 24-48h
4. **Configurer :**
   ```bash
   cp .env.example .env
   echo "ENTSOE_API_KEY=votre_clé_ici" >> .env
   ```

**Tester la connexion :**
```bash
python test_api_connection.py
# Attendu: ✅ ALL TESTS PASSED
```

**Collecter les prix :**
```bash
python data_recuperation/data_market_prices.py \
  --start_date 2023-01-01 \
  --end_date 2024-12-31 \
  --countries FR

# Valider qualité
python data_collection/data_validator.py \
  data/raw_data/market_prices/day_ahead_prices_FR.csv
```

### 4. Fondamentaux (ENTSO-E) - Même clé API 🔑

```bash
# Génération, load, flux transfrontaliers
python data_recuperation/data_fundamentals.py \
  --start_date 2023-01-01 \
  --end_date 2024-12-31 \
  --countries FR
```

---

## 📊 Workflow Complet (Ordre Recommandé)

### Option A : Commencer MAINTENANT (sans clé API)

```bash
# 1. Météo (10 min, aucune clé)
python data_collection/pipeline.py weather-historical --frequency daily

# 2. Consommation (10 min, aucune clé)
python data_collection/odre_collector.py \
  --start_date 2023-01-01 --end_date 2024-12-31

# ✅ Vous avez 40% des données!
# Vous pouvez déjà explorer les corrélations météo-consommation
```

### Option B : Workflow Complet (avec clé API)

```bash
# 1. Configuration
cp .env.example .env
# Ajouter: ENTSOE_API_KEY=votre_clé

# 2. Tout collecter (1-2h total)
python data_collection/pipeline.py weather-historical --frequency daily
python data_collection/odre_collector.py --start_date 2023-01-01 --end_date 2024-12-31
python data_recuperation/data_market_prices.py --start_date 2023-01-01 --end_date 2024-12-31 --countries FR
python data_recuperation/data_fundamentals.py --start_date 2023-01-01 --end_date 2024-12-31 --countries FR

# 3. Vérifier
ls -lh data/raw_data/weather/
ls -lh data/raw_data/energy/
ls -lh data/raw_data/market_prices/
ls -lh data/raw_data/fundamentals/

# ✅ 100% des données collectées!
```

---

## 🎯 Récapitulatif Ultra-Rapide

| Action | Temps | Clé API? | Commande |
|--------|-------|----------|----------|
| **Météo** | 10 min | ❌ Non | `python data_collection/pipeline.py weather-historical` |
| **Consommation** | 10 min | ❌ Non | `python data_collection/odre_collector.py --start_date 2023-01-01 --end_date 2024-12-31` |
| **Prix électricité** | 30 min | ✅ ENTSO-E | `python data_recuperation/data_market_prices.py --start_date 2023-01-01 --end_date 2024-12-31 --countries FR` |
| **Fondamentaux** | 30 min | ✅ ENTSO-E | `python data_recuperation/data_fundamentals.py --start_date 2023-01-01 --end_date 2024-12-31 --countries FR` |

**Total avec clé API :** 1h20 de collecte
**Total sans clé API :** 20 min de collecte (40% des données)

---

## 🔍 Vérification

### Vérifier que tout est bien collecté

```bash
# Météo
ls data/raw_data/weather/*.csv
# Attendu: weather_daily_*.csv, weather_forecast_*.csv

# Consommation
ls data/raw_data/energy/*.csv
# Attendu: odre_consumption.csv

# Prix (nécessite clé ENTSO-E)
ls data/raw_data/market_prices/*.csv
# Attendu: day_ahead_prices_FR.csv

# Fondamentaux (nécessite clé ENTSO-E)
ls data/raw_data/fundamentals/*.csv
# Attendu: load_FR.csv, generation_FR.csv
```

### Valider la qualité

```bash
# Valider prix
python data_collection/data_validator.py \
  data/raw_data/market_prices/day_ahead_prices_FR.csv --type prices

# Valider consommation (load)
python data_collection/data_validator.py \
  data/raw_data/fundamentals/load_FR.csv --type load
```

---

## 🚨 Troubleshooting

### "API key not found"
```bash
# Vérifier .env existe
ls -la .env

# Vérifier contenu
cat .env | grep ENTSOE_API_KEY
# Devrait afficher: ENTSOE_API_KEY=votre_clé_ici (pas d'espaces!)
```

### "Invalid API key"
- Clé pas encore activée (attendre 24-48h)
- Vérifier sur https://transparency.entsoe.eu/ que la clé est bien générée
- Essayer de régénérer une nouvelle clé

### "Rate limit reached"
- Normal ! Le connecteur attend automatiquement
- Pour 3 ans de données, ça peut prendre 30-60 min
- Utilisez le cache pour éviter de re-télécharger

### Vérifier le cache
```bash
# Statistiques cache
python data_collection/api_cache.py --stats

# Nettoyer cache ancien (>30 jours)
python data_collection/api_cache.py --clear 30
```

---

## 📖 Documentation Complète

- **Setup complet ENTSO-E :** `docs/ENTSOE_API_SETUP.md`
- **Migration vers données réelles :** `docs/MIGRATION_GUIDE.md`
- **Audit sources de données :** `AUDIT_DATA_SOURCES.md`
- **Configuration :** `config.yaml`

---

## 🎉 Résultat Final

**Avec juste 1 clé API (gratuite, 5 min pour obtenir), vous avez accès à :**

✅ Prix électricité day-ahead (ENTSO-E)
✅ Données météo historiques + prévisions (Open-Meteo)
✅ Consommation électricité + gaz par région (ODRE)
✅ Génération par type, load, flux (ENTSO-E)

**4 sources de données professionnelles**
**3 fonctionnent sans clé**
**1 seule clé API nécessaire (ENTSO-E)**

**Le projet est 100% prêt pour les données réelles ! 🚀**
