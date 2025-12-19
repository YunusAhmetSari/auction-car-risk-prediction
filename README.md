# Predicting risky auction car purchases

### Executive summary (EN)

- **Problem:** Auctions force split-second decisions; misjudgements lead to write-offs and lost capital.
- **Solution:** Supervised classification for predicting `IsBadBuy` based on vehicle, price and dealer characteristics.
- **Results:** XGBoost model achieves 62% recall for "bad buys" (target: maximize recall) with 26% precision, 37% F1-score, and 74% overall accuracy.
- **Impact:** Better risk management before bidding; focus on recall for "bad buys" to avoid expensive mispurchases.

# Auction Car Risk Prediction

> Vorhersage des Ausfallrisikos von Auktionseinkäufen, um Händlern datenbasierte Biet- und Ankaufentscheidungen zu ermöglichen.

## Projektübersicht

**Problemstellung**  
Im Großhandel mit Gebrauchtwagen fehlen belastbare Echtzeitindikatoren für versteckte Mängel. Fehlentscheidungen verursachen Nacharbeit, Garantiekosten und Kapitalbindung.

**Ziel**  
Maximierung des Recalls für "Bad Buys" (`IsBadBuy=1`), um möglichst viele problematische Fahrzeuge zu identifizieren, bei gleichzeitig kontrollierter Precision, um nicht zu viele gute Käufe fälschlicherweise als schlecht zu klassifizieren.

**Methoden**

- **Preprocessing:** Datentypkonvertierung, Imputation (Median für numerisch, Most Frequent/Constant für kategorial), Feature Engineering (CostPerMile, WarrantyPerCost, MilesPerYear)
- **Feature Selection:** Entfernung redundanter Features (PurchDate, VehYear, WheelTypeID, etc.)
- **Encoding:** One-Hot-Encoding für kategoriale Features, Top-N-Bucketing für hochkardinale Features (TopNCategoriesTransformer)
- **Dimensionality Reduction:** PCA (95% Varianz) für MMR-Preisfeatures
- **Class Imbalance:** RandomUnderSampler
- **Modelle:** RandomForest, LogisticRegression, DecisionTree, KNeighbors, XGBoost
- **Hyperparameter-Optimierung:** BayesSearchCV für XGBoost (100 Iterationen, 5-Fold Stratified CV, F1-Score als Metrik)
- **Model Interpretation:** Feature Importance, Partial Dependence Plots (PDP) für VehicleAge und VehOdo

## Daten

- **Quelle:** Kaggle Competition [Don’t Get Kicked!](https://www.kaggle.com/competitions/DontGetKicked/data)
- **Größe:** 72 Features, 72 983 Fahrzeuge, Zielvariable `IsBadBuy`
- **Besonderheiten:** Unausgewogene Klassen (~13 % Bad Buys), heterogene Datenqualität

## Arbeitsablauf (Workflow)

1. **Datenakquisition:** Automatischer Download von Kaggle Competition via `kagglehub`
2. **Explorative Datenanalyse (EDA):** Verteilungen, Korrelationen, fehlende Werte, Outlier-Erkennung
3. **Data Preparation:**
   - Datentypkonvertierung und -bereinigung
   - Feature Engineering (neue Features: CostPerMile, WarrantyPerCost, MilesPerYear)
   - Feature Selection (Entfernung redundanter/irrelevanter Features)
4. **Preprocessing Pipeline:**
   - Imputation (numerisch: Median, kategorial: Most Frequent/Constant)
   - Encoding (One-Hot-Encoding mit Top-N-Bucketing für hochkardinale Features)
   - Skalierung (StandardScaler)
   - PCA für MMR-Preisfeatures
5. **Modellierung:**
   - Baseline-Modelle (RandomForest, LogisticRegression, DecisionTree, KNeighbors, XGBoost)
   - Hyperparameter-Optimierung mit BayesSearchCV (XGBoost)
   - Evaluation auf Test-Set (Precision, Recall, F1-Score)
6. **Model Interpretation:**
   - Feature Importance (gruppiert nach Original-Features)
   - Partial Dependence Plots für VehicleAge und VehOdo
   - PDP-Interaktion zwischen VehicleAge und VehOdo
7. **Modell-Persistierung:** Gespeicherte Pipeline für spätere Vorhersagen

## Erkenntnisse aus der bisherigen Analyse

**Modell-Performance:**

- **Bestes Modell:** XGBoost mit optimierten Hyperparametern
- **Test-Metriken (IsBadBuy=1):**
  - Recall: **0.62** (62% der "Bad Buys" werden erkannt)
  - Precision: 0.26 (26% der als "Bad Buy" klassifizierten sind tatsächlich schlecht)
  - F1-Score: 0.37
- **Test-Metriken (IsBadBuy=0):**
  - Recall: 0.76
  - Precision: 0.93
  - F1-Score: 0.84
- **Gesamt-Accuracy:** 0.74

**Wichtige Features (Top 20):**

- VehicleAge, VehOdo, VehBCost, WarrantyCost
- MMR-Preisfeatures (via PCA)
- Kategoriale Features: Make, Model, Color, Size, VNST, Trim, SubModel

**Partial Dependence Erkenntnisse:**

- **VehicleAge:** Ältere Fahrzeuge zeigen höheres Risiko für "Bad Buys"
- **VehOdo:** Höhere Kilometerstände korrelieren mit erhöhtem Risiko
- **Interaktion:** Kombination aus hohem Alter und hohem Kilometerstand verstärkt das Risiko

**Business Impact:**

- Das Modell identifiziert 62% aller problematischen Fahrzeuge (Recall)
- Bei 26% Precision bedeutet dies, dass etwa 3 von 4 als "risky" klassifizierten Fahrzeugen tatsächlich problematisch sind
- Fokus auf Recall ist für das Geschäftsziel (Vermeidung teurer Fehlkäufe) optimal

## Reproduzierbarkeit

```bash
# Repository klonen
git clone https://github.com/YunusAhmetSari/auction-car-risk-prediction.git
cd auction-car-risk-prediction

# Dependencies installieren
uv sync
```

**Kaggle API konfigurieren**

1. `kaggle.json` unter `C:\Users\username\.kaggle\kaggle.json` speichern
2. Competition-Regeln auf Kaggle akzeptieren

## Repository Struktur

```
├── data/
│   ├── raw/                # Original-Kaggle-Dateien (ignored)
│   └── processed/          # Gespeicherte Modelle (best_model_pipeline.pkl)
├── notebooks/
│   └── auction_car_risk_pipeline.ipynb  # Vollständiger Workflow (EDA → Modeling → Interpretation)
├── src/core/
│   └── data.py             # Utility-Funktionen (Kaggle-Download, Cleaning, Feature Engineering, TopNCategoriesTransformer)
└── README.md
```

## Über dieses Projekt

- **Kontext:** Data-Science-Projekt
- **Zeitraum:** 25.11.2025 - 19.12.2025
- **Autor:** Yunus Ahmet Sari

## Kontakt

**[GitHub](https://github.com/YunusAhmetSari)** | **[LinkedIn](https://www.linkedin.com/in/yunussari/)** | **[E-Mail](mailto:yunusahmet61@gmail.com)**
