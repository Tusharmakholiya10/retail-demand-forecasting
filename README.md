# 📈 Demand Forecasting & Inventory Optimization System

An end-to-end **Machine Learning demand forecasting system** that predicts future store-product demand and converts those predictions into actionable **inventory and reorder recommendations**.

The project uses historical retail sales data, time-series features, promotions, holidays, oil prices, store information, and a tuned **LightGBM** model to forecast demand.

It also includes a **Streamlit dashboard** for visualization, forecast analysis, model evaluation, and inventory recommendations.

---

## 🚀 Project Overview

Retail businesses need accurate demand forecasts to maintain the right amount of inventory.

Too much inventory can increase:

- Storage costs
- Waste
- Capital tied up in stock

Too little inventory can cause:

- Stockouts
- Lost sales
- Poor customer experience

This project addresses the problem using a complete ML pipeline:

```text
Raw Retail Data
       ↓
Data Cleaning
       ↓
External Data Preparation
       ↓
Feature Engineering
       ↓
Time-Series Train/Validation Split
       ↓
Baseline Models
       ↓
LightGBM / XGBoost
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Future Demand Forecast
       ↓
Inventory Recommendation
       ↓
Streamlit Dashboard