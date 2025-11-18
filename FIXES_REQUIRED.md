# Corrections Requises - Plan d'Action

## 🔴 PRIORITÉ 1: Corriger le Data Leakage (CRITIQUE)

### Fichier: `scripts/prepare_training_data.py`

#### Changement 1: Modifier l'ordre des opérations dans `main()`

**Ligne 229-241 (actuel - INCORRECT):**
```python
try:
    # Load data
    df_market = load_market_data()
    df_weather = load_weather_data(frequency=args.frequency)

    # Merge
    df_merged = merge_all_data(df_market, df_weather, frequency=args.frequency)

    # Engineer features ❌ SUR TOUT LE DATASET
    df_features = engineer_features(df_merged)

    # Split ❌ APRÈS feature engineering
    df_train, df_test = split_train_test(df_features, test_size=args.test_size)
```

**REMPLACER PAR (CORRECT):**
```python
try:
    # Load data
    df_market = load_market_data()
    df_weather = load_weather_data(frequency=args.frequency)

    # Merge
    df_merged = merge_all_data(df_market, df_weather, frequency=args.frequency)

    # Split AVANT feature engineering ✅
    df_train_raw, df_test_raw = split_train_test(df_merged, test_size=args.test_size)

    # Engineer features SÉPARÉMENT ✅
    logger.info("\nEngineering features on TRAIN set...")
    df_train = engineer_features(df_train_raw)

    logger.info("\nEngineering features on TEST set...")
    df_test = engineer_features(df_test_raw)
```

#### Changement 2: Vérifier que `engineer_features()` n'a pas de leakage

La fonction `engineer_features()` est correcte SI ET SEULEMENT SI elle est appelée séparément sur train et test.

Les lags et rolling means utilisent déjà `shift()`, donc ils sont corrects:
```python
# Ces lignes sont OK (utilisent le passé uniquement)
df[f'load_lag_{lag}'] = df['load_mw'].shift(lag)  # ✅
df[f'load_rolling_mean_{window}'] = df['load_mw'].shift(1).rolling(window).mean()  # ✅
```

**Pas de modification requise dans `engineer_features()` SI on l'appelle après le split.**

---

## 🔴 PRIORITÉ 2: Corriger les Métriques (HAUTE)

### Fichier: `scripts/run_trading_inference.py`

#### Changement 1: Corriger le calcul de l'annual return

**Ligne 307 (actuel - INCORRECT):**
```python
annual_return = (1 + total_return) ** (252 / len(returns)) - 1
```

**REMPLACER PAR:**
```python
# Use 365 days for electricity markets (trades every day)
annual_return = (1 + total_return) ** (365 / len(returns)) - 1
```

#### Changement 2: Corriger le calcul du Sharpe ratio

**Ligne 309 (actuel - INCORRECT):**
```python
sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
```

**REMPLACER PAR:**
```python
# Use 365 days for electricity markets (trades every day, not just business days)
sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
```

#### Changement 3: Ajouter un commentaire explicatif

Ajouter ce commentaire au début de la fonction `run_backtest()`:
```python
def run_backtest(market_prices, price_forecasts, dates, model_name, initial_capital=100000):
    """
    Run realistic electricity trading backtest.

    IMPORTANT: Electricity markets trade 365 days/year (not 252 like equity markets).
    Therefore, we use 365 for annualization factors in Sharpe ratio and returns.

    Model: Simplified spread trading
    ...
    """
```

---

## 🟡 PRIORITÉ 3: Améliorer la Validation (RECOMMANDÉ)

### Objectif
Augmenter la robustesse de la validation en testant sur plusieurs périodes.

### Option A: Walk-Forward Validation (Recommandé)

Créer un nouveau fichier: `scripts/run_walk_forward_validation.py`

```python
"""
Walk-Forward Validation for Trading Strategies

Tests the strategy on multiple rolling windows to ensure robustness.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def walk_forward_validation(df, model, window_size=180, test_size=60):
    """
    Perform walk-forward validation.

    Args:
        df: Full dataset
        model: Trained model
        window_size: Training window size (days)
        test_size: Test window size (days)

    Returns:
        list: Results for each fold
    """
    results = []

    # Create rolling windows
    n_windows = (len(df) - window_size) // test_size

    for i in range(n_windows):
        train_start = i * test_size
        train_end = train_start + window_size
        test_start = train_end
        test_end = test_start + test_size

        if test_end > len(df):
            break

        df_train = df.iloc[train_start:train_end]
        df_test = df.iloc[test_start:test_end]

        # Retrain model on this window
        # ... (code to retrain)

        # Backtest on this window
        # ... (code to backtest)

        results.append({
            'fold': i,
            'train_period': f"{df_train['datetime'].min()} to {df_train['datetime'].max()}",
            'test_period': f"{df_test['datetime'].min()} to {df_test['datetime'].max()}",
            'sharpe': sharpe,
            'total_return': total_return,
            # ... autres métriques
        })

    return results
```

### Option B: Augmenter le Test Set (Plus Simple)

Modifier le split 80/20 → 60/40 pour avoir ~1 an de test:

Dans `scripts/prepare_training_data.py`:
```python
parser.add_argument(
    "--test-size",
    type=float,
    default=0.4,  # Changed from 0.2 to 0.4
    help="Test set size (default: 0.4)"
)
```

---

## 🔍 PRIORITÉ 4: Vérifier la Standardisation (À VÉRIFIER)

### Fichier: `data_processing/transformation.py`

#### Vérification requise

Vérifier que le scaler est UNIQUEMENT fit sur le train set:

**Scénario A (CORRECT):**
```python
# Lors de l'entraînement
df_train_transformed = transform_regression_and_xgb(
    df_train,
    fit_scaler=True,  # ✅ Fit sur train
    save=True
)

# Lors du test
df_test_transformed = transform_regression_and_xgb(
    df_test,
    fit_scaler=False,  # ✅ Utilise le scaler sauvegardé
    save=False
)
```

**Scénario B (INCORRECT - DATA LEAKAGE):**
```python
# Si on transforme tout le dataset ensemble
df_all = pd.concat([df_train, df_test])
df_all_transformed = transform_regression_and_xgb(
    df_all,
    fit_scaler=True,  # ❌ FIT SUR TRAIN+TEST
    save=True
)
```

**Action:** Vérifier dans les scripts d'entraînement (`train_simple_model.py`) comment la transformation est appelée.

**Si la transformation est appelée sur tout le dataset:** Corriger en appelant séparément sur train et test.

---

## 📋 Checklist de Validation Post-Corrections

Après avoir appliqué les corrections, vérifier:

### ✅ Data Integrity
- [ ] Le split train/test se fait AVANT le feature engineering
- [ ] Les features sont créées séparément sur train et test
- [ ] Le scaler est fit uniquement sur train
- [ ] Aucune feature ne "voit" le futur

### ✅ Metrics Correctness
- [ ] Le Sharpe ratio utilise sqrt(365)
- [ ] L'annual return utilise 365 jours
- [ ] Les métriques sont calculées sur une période > 6 mois

### ✅ Results Sanity Check
- [ ] Sharpe ratio entre 0.5 et 1.5 (réaliste pour trading)
- [ ] R² entre 0.30 et 0.50 (réaliste pour prix électricité)
- [ ] Win rate entre 50% et 60%
- [ ] Max drawdown entre -10% et -20%
- [ ] Pas de trades avec PnL > 10% du capital en 1 jour

### ✅ Code Quality
- [ ] Commentaires explicatifs sur le data leakage
- [ ] Tests unitaires pour vérifier l'absence de leakage
- [ ] Documentation mise à jour

---

## 🎯 Plan d'Exécution (Étape par Étape)

### Jour 1: Corrections Critiques
1. ✅ Corriger `prepare_training_data.py` (split avant features)
2. ✅ Corriger `run_trading_inference.py` (métriques)
3. ✅ Vérifier `transformation.py` (scaler)
4. ✅ Commit les changements

### Jour 2: Régénération des Données
1. ✅ Supprimer les anciennes données train/test
2. ✅ Relancer `prepare_training_data.py` avec les corrections
3. ✅ Vérifier que les fichiers générés sont corrects
4. ✅ Analyser les nouveaux datasets (stats descriptives)

### Jour 3: Réentraînement des Modèles
1. ✅ Supprimer les anciens modèles
2. ✅ Relancer `train_simple_model.py` pour tous les modèles
3. ✅ Comparer les nouveaux R² avec les anciens
4. ✅ Documenter la baisse de performance (attendue)

### Jour 4: Backtests
1. ✅ Relancer `run_trading_inference.py` pour tous les modèles
2. ✅ Analyser les nouveaux Sharpe ratios
3. ✅ Comparer avec les résultats biaisés
4. ✅ Créer un rapport de comparaison

### Jour 5: Validation et Documentation
1. ✅ (Optionnel) Implémenter walk-forward validation
2. ✅ Créer un rapport final avec tous les résultats
3. ✅ Mettre à jour le README avec les vrais résultats
4. ✅ Présenter les résultats aux stakeholders

---

## 📊 Résultats Attendus

### Avant Corrections (Biaisés)
| Modèle | R² | Sharpe | Return Annuel | Win Rate |
|--------|-----|--------|---------------|----------|
| Random Forest | 0.641 | 2.22 | 54.9% | 61.3% |
| XGBoost | 0.686 | 1.95 | 47.9% | 57.6% |
| LightGBM | 0.678 | 1.59 | 38.2% | 55.2% |
| Ridge | 0.437 | 1.00 | 14.0% | 63.3% |

### Après Corrections (Attendu)
| Modèle | R² | Sharpe | Return Annuel | Win Rate |
|--------|-----|--------|---------------|----------|
| Random Forest | 0.35-0.45 | 0.8-1.2 | 15-25% | 52-58% |
| XGBoost | 0.40-0.50 | 0.7-1.1 | 12-22% | 51-56% |
| LightGBM | 0.38-0.48 | 0.6-1.0 | 10-20% | 50-55% |
| Ridge | 0.25-0.35 | 0.4-0.7 | 5-12% | 50-54% |

**Note:** Ces prédictions sont conservatrices. Les vrais résultats peuvent être meilleurs ou pires.

---

## 💡 Recommandations Finales

1. **Ne pas paniquer:** Une baisse de performance est normale et attendue après correction du data leakage

2. **Sharpe 0.8-1.2 est EXCELLENT pour du trading d'électricité:**
   - C'est un marché difficile à prédire
   - R² de 0.4 signifie qu'on explique 40% de la variance (très bon)
   - Un Sharpe de 1.0 est commercialement viable

3. **Améliorer le modèle après corrections:**
   - Ajouter plus de features (prix du gaz, météo régionale)
   - Essayer des modèles plus sophistiqués (LSTM, Transformers)
   - Optimiser les hyperparamètres avec Optuna
   - Implémenter des ensembles de modèles

4. **Validation continue:**
   - Mettre en place un monitoring en production
   - Réentraîner régulièrement (tous les mois)
   - Surveiller la dégradation du modèle (concept drift)

---

## 🔗 Références

- [Cross-validation for time series](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Data leakage in machine learning](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Sharpe ratio for trading strategies](https://www.investopedia.com/terms/s/sharperatio.asp)
- [Walk-forward validation](https://en.wikipedia.org/wiki/Walk_forward_optimization)
