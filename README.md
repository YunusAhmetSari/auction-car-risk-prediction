# Predicting risky auction car purchases

### Executive summary (EN)

- **Problem:** Auctions force split-second decisions; misjudgements lead to write-offs and lost capital.
- **Solution:** Supervised classification for predicting `IsBadBuy` based on vehicle, price and dealer characteristics.
- **Results:**
- **Impact:** Better risk management before bidding; focus on recall for "bad buys" to avoid expensive mispurchases.

# Auction Car Risk Prediction

> Vorhersage des Ausfallrisikos von Auktionseinkäufen, um Händlern datenbasierte Biet- und Ankaufentscheidungen zu ermöglichen.

## Projektübersicht

**Problemstellung**  
Im Großhandel mit Gebrauchtwagen fehlen belastbare Echtzeitindikatoren für versteckte Mängel. Fehlentscheidungen verursachen Nacharbeit, Garantiekosten und Kapitalbindung.

**Ziel**

**Methoden**

## Daten

- **Quelle:** Kaggle Competition [Don’t Get Kicked!](https://www.kaggle.com/competitions/DontGetKicked/data)
- **Größe:** 72 Features, 72 983 Fahrzeuge, Zielvariable `IsBadBuy`
- **Besonderheiten:** Unausgewogene Klassen (~13 % Bad Buys), heterogene Datenqualität

## Arbeitsablauf (Workflow)

WIP

## Erkenntnisse aus der bisherigen Analyse

WIP

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
│   └── processed/          # Features, Splits, Artefakte
├── notebooks/
│   └── 01_exploration.ipynb
├── src/core/
│   └── data.py             # Kaggle-Helper
├── docs/
│   └── project.md
└── README.md
```

## Über dieses Projekt

- **Kontext:** Data-Science-Projekt
- **Zeitraum:** 25.11.2025 - tbd
- **Autor:** Yunus Ahmet Sari

## Kontakt

**[GitHub](https://github.com/YunusAhmetSari)** | **[LinkedIn](https://www.linkedin.com/in/yunussari/)** | **[E-Mail](mailto:yunusahmet61@gmail.com)**
