# Audit Approfondi du Pipeline ML - Energy Demand Forecast

## Verdict Global

**Score : 6.5/10 pour un profil junior quant research**

Le projet montre une bonne compréhension des concepts ML et de la structure d'un pipeline de trading quantitatif. Cependant, plusieurs failles méthodologiques critiques compromettent la crédibilité des résultats affichés (Sharpe 1.55). Ce document détaille chaque point.

---

## 1. ANALYSE DES DATA LEAKS ET BIAIS

### 1.1 CRITIQUE - Futures synthétiques (FLAW MAJEUR)

**Fichier : `data_collection/futures_data.py:87-89`**

```python
df['spot_forward_ma'] = df[price_col].rolling(window, min_periods=1).mean().shift(-window)
df['spot_forward_ma'] = df['spot_forward_ma'].fillna(method='bfill')
```

**C'est un look-ahead bias DIRECT.** La construction des futures utilise `shift(-window)` qui regarde 21 jours dans le futur du spot. Le modèle trade sur des futures construits à partir de prix spot futurs qu'il ne pouvait pas connaître au moment du trade.

**Impact :** Le Sharpe Ratio de 1.55 est artificiellement gonflé. Sur des données EEX réelles, on attendrait 0.3-0.8 au mieux.

**Sévérité : ÉLIMINATOIRE** en entretien quant. Un interviewer senior détecterait ce biais en 30 secondes.

### 1.2 Rolling features sans shift correct

**Fichier : `model/price_forecasting/data_loader.py:475-487`**

```python
df[f"{target_col}_roll_mean_{window}h"] = (
    df[target_col].rolling(window=window, min_periods=1).mean()
)
```

Les rolling statistics (mean, std, min, max) sur le target `price` **incluent le point courant**. En production, au moment t, on ne connaît pas `price[t]`, seulement `price[t-1]` et avant.

**Correction nécessaire :** Ajouter `.shift(1)` après chaque `.rolling()` sur le target.

```python
df[f"{target_col}_roll_mean_{window}h"] = (
    df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
)
```

**Sévérité : HAUTE** - Data leak subtil qui gonfle les métriques de forecasting.

### 1.3 Features load_mw - Bon reflexe, mais incomplet

**Fichier : `data_loader.py:536-541`**

```python
feature_cols = [
    col for col in df.columns
    if col not in ["datetime_hour", target_col, "load_mw"]
]
```

L'exclusion de `load_mw` contemporain est correcte. Les lags de load sont utilisés à la place. **BON POINT.**

Cependant, les mêmes rolling features sur `load_mw` (sans shift) ont le même problème qu'au §1.2.

### 1.4 Fuel prices - Leak potentiel

**Fichier : `data_loader.py:622-633`**

Les fuel prices contemporains (ttf_gas, eua_carbon, coal, spreads) sont exclus des features. Cependant, `prepare_price_forecasting_with_fuel_prices` appelle `generate_fuel_price_features()` avec les `power_prices` en argument. Si les fuel prices sont générés à partir des prix spot (corrélation synthétique), c'est un **leak indirect**.

### 1.5 Pas de test leakage au sens strict

Le walk-forward validator (`walk_forward_validator.py:203-219`) est correctement implémenté :
- Train uniquement sur données passées (`df.iloc[:current_idx]`)
- Predict sur le jour suivant
- Retrain quotidien
- Pas de contamination train→test

**BON POINT.** La structure walk-forward est propre.

---

## 2. OVERFITTING ET ROBUSTESSE STATISTIQUE

### 2.1 Hyperparamètres optimisés sur le backtest (SNOOPING)

**Fichier : `production/trading_pipeline.py:43-52`**

```python
SIGNAL_THRESHOLD_QUANTILE = 0.65
MIN_HOLDING_DAYS = 2
VOL_FILTER_QUANTILE = 0.85
```

Ces paramètres sont **hardcodés** et clairement le résultat d'une optimisation sur l'ensemble du backtest. Le README le confirme : le min holding period a fait passer le Sharpe de 0.19 à 1.55. C'est du **strategy snooping** classique.

**En industrie :** Ces paramètres devraient être optimisés dans un walk-forward séparé, avec une période de validation out-of-sample gelée.

### 2.2 Taille d'échantillon insuffisante

- 285 jours de test, 70 trades
- Intervalles de confiance sur le Sharpe : avec 70 trades, σ(Sharpe) ≈ √(2/70) ≈ 0.17
- IC 95% : [1.55 - 2×0.17, 1.55 + 2×0.17] = **[1.21, 1.89]**
- Mais avec le snooping des hyperparamètres, le vrai Sharpe est probablement **0.3-0.7**

### 2.3 Ratio features/observations

- ~50 features pour ~285×24 = 6840 heures de test
- Ratio acceptable pour les tree-based models (XGBoost, LightGBM)
- Problématique pour Ridge sans régularisation forte

### 2.4 Absence de bootstrap / Monte Carlo

Aucune simulation de robustesse :
- Pas de bootstrap des trades pour IC du Sharpe
- Pas de permutation test (le signal est-il meilleur qu'aléatoire ?)
- Pas de sensitivity analysis sur les paramètres de trading

---

## 3. PIPELINE ML - CONFORMITÉ AUX STANDARDS

### 3.1 Ce qui est BIEN fait

| Composant | Évaluation | Détail |
|-----------|-----------|--------|
| Walk-forward validation | **Correct** | Expanding window, retrain quotidien, pas de leakage structurel |
| Exclusion load contemporain | **Correct** | `load_mw` exclu, seuls les lags sont utilisés |
| Métriques multiples | **Bon** | MAE, RMSE, sMAPE, MASE, direction accuracy |
| Quantile regression | **Bon** | LightGBM quantile pour incertitude |
| Transaction costs | **Présents** | 0.15 EUR/MWh (broker + slippage) |
| Code structure | **Propre** | Séparation data/model/production, classes abstraites |
| Data validation | **Bon** | Checks sur prix simulés vs réels |

### 3.2 Ce qui est MAL fait ou MANQUANT

| Composant | Problème | Sévérité |
|-----------|----------|----------|
| Futures synthétiques | Look-ahead bias dans construction | **CRITIQUE** |
| Rolling features | Pas de shift(1) sur target | **HAUTE** |
| Strategy params | Optimisés sur backtest complet | **HAUTE** |
| Out-of-sample test | Absent (pas de holdout 2025) | **HAUTE** |
| Significance testing | Pas de bootstrap/permutation | **MOYENNE** |
| Ensemble weights | Égaux (1/3, 1/3, 1/3) - pas optimisés | **BASSE** |
| Feature selection | Aucune (pas de SHAP, pas de RFE) | **MOYENNE** |
| Stationarity tests | Pas d'ADF/KPSS sur les séries | **MOYENNE** |
| Cointegration | Pas de test spot-futures | **BASSE** (car futures synthétiques) |

### 3.3 Standards industrie manquants

Pour un desk quant en énergie :

1. **Backtesting framework :** Pas de séparation train/validation/test. En industrie, on gèle les 3-6 derniers mois comme holdout pur.

2. **Model monitoring :** Pas de drift detection, pas de feature importance tracking temporel.

3. **Execution model :** Pas de modélisation du carnet d'ordres, du bid-ask spread dynamique, ni du market impact.

4. **Risk management :** VaR et CVaR mentionnés dans le README mais pas implémentés dans le code.

---

## 4. PERTINENCE POUR UN POSTE JUNIOR QUANT RESEARCH

### 4.1 Forces (ce qui impressionnerait)

1. **Scope ambitieux** : Pipeline end-to-end data→model→trading→PnL
2. **Walk-forward correct** : Montre la compréhension du temporal ordering
3. **Quantile regression** : Approche probabiliste, pas juste point forecast
4. **Bayesian optimization** : Via GP + EI, approche avancée pour le tuning
5. **Ensemble methods** : Combinaison Ridge/XGBoost/LightGBM
6. **Market microstructure** : Signal surprise normalisé par volatilité, regime filters
7. **Code quality** : Classes abstraites, typing, docstrings, modularité

### 4.2 Faiblesses (ce qui serait scruté en entretien)

1. **Futures synthétiques avec look-ahead** : Red flag immédiat. Un interviewer demandera "d'où viennent vos futures ?" et le `shift(-21)` est indéfendable.

2. **Sharpe 1.55 non crédible** : En énergie, un Sharpe > 1.0 sur une stratégie directionnelle simple est suspect. Les desks établis font 0.5-1.0 avec des modèles bien plus sophistiqués.

3. **Pas de feature engineering domain-specific** : Absence de features clés en énergie :
   - Interconnections transfrontalières (FR-DE, FR-ES, FR-GB)
   - Disponibilité nucléaire (critique pour la France)
   - Prévisions éoliennes/solaires granulaires
   - Calendrier des maintenances RTE
   - Flow-based market coupling

4. **Pas de réflexion sur la stationnarité** : Les prix électriques sont mean-reverting avec sauts. Aucun test ADF, pas de décomposition (Weron 2006).

5. **Ridge comme "baseline"** : En quant energy, le baseline serait plutôt un modèle AR(p) ou SARIMA, pas un Ridge sur features tabulaires.

### 4.3 Verdict pour entretien

| Aspect | Score | Commentaire |
|--------|-------|-------------|
| Initiative et scope | 8/10 | Projet ambitieux et bien structuré |
| ML pipeline correctness | 5/10 | WFV bon mais rolling leak et futures biaisées |
| Domain knowledge énergie | 4/10 | Manque features nucléaire, interconnexions |
| Rigueur statistique | 4/10 | Pas de significance tests, snooping |
| Code quality | 7/10 | Bien structuré, quelques imports cassés |
| Trading knowledge | 6/10 | Bons concepts mais exécution naïve |
| **Global** | **6/10** | Bon point de départ, corrections nécessaires |

---

## 5. RECOMMANDATIONS PRIORITAIRES

### Tier 1 - Corrections obligatoires (avant entretien)

1. **Supprimer les futures synthétiques.** Utiliser soit :
   - Des données EEX réelles (academic license ~500 EUR/an)
   - Un spread fixe spot + premium saisonnier SANS look-ahead
   - Reformuler la stratégie en pur day-ahead (pas de futures)

2. **Ajouter `.shift(1)` sur toutes les rolling features du target :**
   ```python
   df[target_col].shift(1).rolling(window).mean()
   ```

3. **Séparer un holdout pur :** Geler les 2-3 derniers mois et ne JAMAIS toucher aux hyperparamètres avec ces données.

4. **Ajouter un permutation test :** Vérifier que le signal bat un signal aléatoire (p < 0.05).

### Tier 2 - Améliorations fortes (pour se démarquer)

5. **Ajouter la disponibilité nucléaire** (données RTE Transparency, gratuites) - c'est LE driver du prix FR.

6. **Baseline SARIMA/AR** : Montrer que le ML bat un modèle classique de séries temporelles.

7. **Bootstrap du Sharpe ratio** : 10,000 resamples pour un IC robuste.

8. **Walk-forward des hyperparamètres** : Optimiser thresholds/holding period en rolling window, pas sur tout le backtest.

### Tier 3 - Nice-to-have

9. Ajouter SHAP values pour l'interprétabilité
10. Tester un modèle Transformer temporal (TFT)
11. Ajouter les interconnexions cross-border
12. Implémenter un vrai carnet d'ordres simplifié

---

## 6. RÉSUMÉ EXÉCUTIF

**Ce projet démontre une bonne maîtrise technique du ML appliqué** (walk-forward, quantile regression, ensemble, Bayesian optimization) mais souffre de **failles méthodologiques classiques de junior** :

1. Le look-ahead dans les futures rend les résultats de trading invalides
2. Les rolling features leakent le target courant
3. Le strategy snooping gonfle artificiellement le Sharpe
4. L'absence de tests statistiques empêche de conclure sur la significativité

**Pour un entretien junior quant :** Le projet montre de l'initiative et une bonne structure. Corriger les 4 points ci-dessus transformerait ce projet de "intéressant mais biaisé" en "solide et rigoureux". La différence entre un candidat qui connaît ces pièges et un qui ne les voit pas est exactement ce que les desks quant évaluent.

**Correction prioritaire absolue :** Remplacer `shift(-21)` dans `futures_data.py:89` par une construction sans look-ahead, et ajouter `.shift(1)` aux rolling features dans `data_loader.py:475-487`.
