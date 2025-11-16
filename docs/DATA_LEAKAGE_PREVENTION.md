# 🛡️ Prévention du Data Leakage - Logique Temporelle

**Ce document explique la logique temporelle critique du projet et comment éviter le data leakage.**

---

## ⚠️ Problème Critique Identifié et Corrigé

**Date:** 2024-11-16
**Sévérité:** CRITIQUE
**Impact:** Performance artificiellement gonflée, modèle inutilisable en production

### Le Problème

Le code initial incluait des **variables contemporaines** dans les features :

```python
# ❌ MAUVAIS (version originale)
Prix_14h = f(Load_14h, Température_14h, Prix_gaz_14h, ...)
             ↑
    On utilise des données du MÊME moment qu'on prédit!
    = IMPOSSIBLE en production (ces valeurs ne sont pas encore connues)
```

**Exemple concret :**
```
Moment: 15 janvier 2024, 12h00

Objectif: Prédire le prix à 14h00

❌ Variables qu'on NE PEUT PAS utiliser:
- load_mw à 14h00 (pas encore connu)
- température réalisée à 14h00 (pas encore mesurée)
- prix gaz à 14h00 (pas encore disponible)

✅ Variables qu'on PEUT utiliser:
- load_mw à 13h00, 12h00, ... (historique)
- température PRÉVUE pour 14h00 (forecast)
- prix gaz à 13h00 (dernière valeur connue)
```

---

## ✅ Solution Implémentée

### Principe Fondamental

**On ne peut utiliser que des données DISPONIBLES au moment de la prédiction.**

### Règle d'Or

```
Prix au temps T = f(Variables au temps T-k)
                      où k > 0 (lags)
```

### Code Corrigé

```python
# ✅ BON (version corrigée)

# Variables EXCLUES des features (contemporaines):
exclude_cols = [
    "datetime_hour",
    "price",           # Target
    "load_mw",         # ❌ Valeur au temps T
    "ttf_gas_price",   # ❌ Valeur au temps T
    "temperature",     # ❌ Valeur réalisée au temps T
    # etc.
]

# Variables INCLUSES (avec lags):
features = [
    "load_mw_lag_1h",    # ✅ T-1
    "load_mw_lag_24h",   # ✅ T-24
    "load_mw_lag_168h",  # ✅ T-168 (1 semaine)
    "ttf_gas_price_lag_24h",  # ✅ Prix gaz d'hier
    "hour", "day_of_week",    # ✅ Features temporelles
    "temperature_forecast",   # ✅ Prévision (si disponible)
]
```

---

## 📊 Exemples de Scénarios

### Scénario 1: Prédiction Day-Ahead (Cas Réel ENTSO-E)

**Contexte:**
- On est le 15/01/2024 à 12h00
- Les prix day-ahead pour le 16/01 sont publiés à 12h30
- On veut prédire les prix du 16/01

**Timeline:**
```
15/01 12:00 ←── On est ICI
    ↓
    Données disponibles:
    - Prix historiques jusqu'au 15/01 12:00
    - Load historique jusqu'au 15/01 11:00 (retard 1h)
    - Météo PRÉVUE pour le 16/01
    - Prix fuel jusqu'au 15/01 (jour J-1)
    ↓
15/01 12:30 ←── Publication prix day-ahead
    ↓
16/01 00:00 ←── Début livraison
```

**Features valides:**
```python
prix_16jan_14h = f(
    prix_lag_24h=prix_15jan_14h,      # ✅ Connu
    load_lag_24h=load_15jan_14h,      # ✅ Connu
    temp_forecast_16jan_14h,          # ✅ Prévision dispo
    hour=14,                           # ✅ Connu
    day_of_week=1,                     # ✅ Connu (mardi)
    ttf_gas_lag_24h=gas_15jan         # ✅ Prix gaz d'hier
)
```

### Scénario 2: Backtest Historique

**Danger:** En backtest, TOUTES les données sont disponibles !

```python
# ❌ PIÈGE FRÉQUENT en backtest:
# On a accès à load_réalisé du 16/01 dans notre dataset
# Mais en production, cette valeur n'existe pas encore!

# Dataset backtest (fichier CSV):
datetime        | price | load_mw | temp_realized
16/01 14:00    | 75    | 68000   | 12.5
                  ↑       ↑         ↑
               Target   ❌ Leak   ❌ Leak

# ✅ CORRECT: Utiliser seulement lags
features = {
    'load_mw_lag_24h': 65000,  # Load du 15/01 14:00
    'temp_forecast': 13.0,      # Prévision météo
}
```

### Scénario 3: Features Météo

**Problème subtil:** Météo réalisée vs. prévisions

```python
# ❌ MAUVAIS:
prix_14h = f(température_réalisée_14h)
# La température réalisée n'est connue qu'APRÈS 14h!

# ✅ BON:
prix_14h = f(température_prévue_14h_faite_à_12h)
# On utilise la prévision météo faite à 12h pour 14h

# 🤔 ACCEPTABLE (avec compromis):
prix_14h = f(température_lag_1h)
# Température d'il y a 1h (bonne proxy pour maintenant)
```

---

## 🔍 Détection du Leakage

### Test Simple

**Si vous pouvez répondre "non" à cette question, il y a leakage :**

> "Au moment où je fais ma prédiction, est-ce que cette variable est déjà connue ?"

### Checklist de Validation

Pour chaque feature, vérifier:

- [ ] **Timestamp de disponibilité:** Quand cette valeur est-elle publiée ?
- [ ] **Retard de publication:** Y a-t-il un délai entre réalisation et publication ?
- [ ] **Source:** Est-ce une valeur réalisée ou une prévision ?
- [ ] **Fréquence de mise à jour:** À quelle fréquence est-elle actualisée ?

### Exemples par Type de Données

| Variable | Disponibilité | Lag Nécessaire | Pourquoi |
|----------|---------------|----------------|----------|
| **Prix day-ahead** | J-1 à 12h30 | 12h30 avant livraison | Publication ENTSO-E |
| **Load réalisé** | J+1 | 24h+ | Consolidation données |
| **Météo réalisée** | Temps réel | ~15min | Mesure stations |
| **Météo prévue** | Temps réel | 0 | Modèles météo |
| **Prix fuel (TTF)** | J à 17h | Fin journée | Publication marché |
| **Génération renouvelable** | Temps réel | ~15min | Télémétrie RTE |

---

## 📈 Impact sur la Performance

### Performance Attendue

**Avec leakage (❌ invalide) :**
```
R² score: 0.95-0.98  ← Artificiel!
MAE: 2-5 EUR/MWh     ← Trop beau pour être vrai
```

**Sans leakage (✅ réaliste) :**
```
R² score: 0.50-0.70  ← Réaliste
MAE: 10-20 EUR/MWh   ← Normal pour prix électricité
```

**Pourquoi cette différence ?**

Avec leakage :
```
Prix_14h = 0.9 * Load_14h + bruit
              ↑
    Corrélation quasi-parfaite (load drive price)
    Mais load_14h n'est pas connu à l'avance!
```

Sans leakage :
```
Prix_14h = 0.6 * Load_lag_24h + 0.3 * Temp_forecast + bruit
              ↑                    ↑
    Corrélation plus faible (prédiction incertaine)
    Mais réaliste et utilisable!
```

---

## 🛠️ Correction Appliquée

### Fichiers Modifiés

**`model/price_forecasting/data_loader.py`**

#### Fonction `prepare_price_forecasting_dataset()`

```python
# AVANT (bugué):
feature_cols = [
    col for col in df.columns
    if col not in ["datetime_hour", target_col]
]
# ➜ Incluait load_mw ❌

# APRÈS (corrigé):
feature_cols = [
    col for col in df.columns
    if col not in ["datetime_hour", target_col, "load_mw"]
]
# ➜ Exclut load_mw, garde seulement les lags ✅
```

#### Fonction `prepare_price_forecasting_with_fuel_prices()`

```python
# AVANT (bugué):
feature_cols = [
    col for col in df.columns
    if col not in ["datetime_hour", target_col]
]
# ➜ Incluait ttf_gas_price, load_mw, etc. ❌

# APRÈS (corrigé):
exclude_cols = [
    "datetime_hour", target_col,
    "load_mw",           # Contemporain ❌
    "ttf_gas_price",     # Contemporain ❌
    "eua_carbon_price",  # Contemporain ❌
    "spark_spread",      # Contemporain ❌
    # etc.
]
feature_cols = [col for col in df.columns if col not in exclude_cols]
# ➜ Garde seulement les lags ✅
```

---

## ✅ Validation

### Tests à Effectuer

**1. Vérifier les features:**
```python
from model.price_forecasting.data_loader import prepare_price_forecasting_dataset

df, features = prepare_price_forecasting_dataset()

# ❌ NE DOIT PAS contenir:
assert "load_mw" not in features
assert "ttf_gas_price" not in features

# ✅ DOIT contenir:
assert "load_mw_lag_24h" in features
assert "price_lag_24h" in features
```

**2. Vérifier la logique temporelle:**
```python
# Simulation: prédire T avec données T-k
df_train = df[df['datetime_hour'] < '2024-01-15']
df_test = df[df['datetime_hour'] >= '2024-01-15']

# Les features de test ne doivent utiliser QUE des données < 2024-01-15
for idx, row in df_test.iterrows():
    # Vérifier que load_mw_lag_24h correspond bien à T-24
    assert row['load_mw_lag_24h'] == df.loc[idx-24, 'load_mw']
```

**3. Comparer performances:**
```python
# Entraîner modèle sans leakage
model_clean = train_model(features_sans_leakage)
score_clean = model_clean.score(X_test, y_test)

# Si score > 0.9, SUSPECT! (probable leakage résiduel)
assert score_clean < 0.8, "Performance suspecte, vérifier leakage"
```

---

## 📚 Ressources

**Lectures recommandées:**
- [Kaggle: Data Leakage Guide](https://www.kaggle.com/code/alexisbcook/data-leakage)
- [Avoiding Data Leakage in Time Series](https://medium.com/@keshavkaul/avoiding-data-leakage-in-time-series-forecasting-8b5c8b6e7f2e)

**Articles académiques:**
- "Common pitfalls in time series forecasting" (Hyndman & Athanasopoulos)
- "Leakage in data mining: Formulation, detection, and avoidance" (Kaufman et al.)

---

## 🎯 Résumé

**Règle d'or:**
> "Utilise seulement ce que tu saurais en production au moment de prédire."

**En pratique:**
- ✅ Lags (T-1, T-24, T-168)
- ✅ Features temporelles (hour, day_of_week)
- ✅ Prévisions météo
- ✅ Rolling statistics sur historique
- ❌ Valeurs contemporaines (T)
- ❌ Valeurs futures (T+1)

**Impact de la correction:**
- Performance va BAISSER (de ~0.95 à ~0.65 R²)
- C'est **NORMAL** et **ATTENDU**
- Le modèle est maintenant **UTILISABLE** en production

**Crédit:** Bug identifié grâce à la question critique de l'utilisateur : "Mais donc la si on réfléchis a la logique derriere cest tout bon? par de probleme de bias avec des données quon devrait pas avoir etc..."

**Bravo d'avoir posé cette question !** 🎉
