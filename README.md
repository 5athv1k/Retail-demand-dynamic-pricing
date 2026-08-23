# Retail Demand Forecasting & Dynamic Pricing Optimization

An end-to-end data science project that combines demand forecasting, price elasticity analysis, and constrained dynamic pricing optimization to support revenue-focused retail decision making.

## 📌 Project Overview

Retail businesses need to answer two important questions:

1. How much demand should we expect in the future?
2. What price is likely to maximize revenue without making unrealistic pricing changes?

This project builds a complete analytics pipeline to address both problems using historical food-ordering data.

## 🎯 Objectives

- Forecast future demand using machine learning
- Benchmark the forecasting model against naive and linear baselines
- Estimate price sensitivity using controlled elasticity analysis
- Identify revenue-maximizing price scenarios
- Apply realistic ±20% pricing constraints
- Evaluate pricing scenarios on a historical holdout period

## 📊 Dataset

**Food Demand Forecasting Dataset — Kaggle**

The project uses:

- `train.csv`
- `meal_info.csv`
- `fulfilment_center_info.csv`

The data contains historical demand, meal information, fulfilment-center information, pricing, and promotion indicators.

## 🧠 Methodology

### 1. Data Preparation

- Merge transactional and lookup datasets
- Check missing values
- Check duplicates
- Validate price values
- Investigate unusual price/base-price relationships

### 2. Exploratory Data Analysis

Analyze:

- Demand distribution
- Price distribution
- Promotion effects
- Category-level demand
- Price-demand relationships
- Revenue patterns

### 3. Feature Engineering

Create forecasting features including:

- Lagged demand
- Rolling demand statistics
- Price-related features
- Promotion indicators
- Category/cuisine/center features
- Time-based features

### 4. Demand Forecasting

Models evaluated:

- Naive last-week baseline
- Ridge regression
- XGBoost

A time-based validation strategy is used so future information does not leak into model training.

### 5. Price Elasticity

Estimate controlled log-log price elasticity while accounting for observed differences across:

- Meal
- Fulfilment center
- Promotion indicators

Elasticity results are interpreted as observational/model-based relationships rather than causal experimental effects.

### 6. Dynamic Pricing Optimization

The optimizer:

- Uses controlled price elasticity
- Searches candidate prices within historical price ranges
- Applies a maximum ±20% price-change constraint
- Estimates demand and expected revenue under each scenario

### 7. Pricing Scenario Evaluation

Pricing recommendations are evaluated on a historical holdout period using forecasted demand rather than assuming the observed future demand would remain unchanged.

## 📈 Key Results

### Demand Forecasting

| Model | MAE |
|---|---:|
| Naive baseline | 99.1 |
| Ridge regression | 113.3 |
| XGBoost | 77.4 |

XGBoost achieved:

- **21.9% lower MAE than the naive baseline**
- **31.7% lower MAE than the Ridge baseline**

### Pricing Optimization

The constrained pricing scenario estimated approximately:

**10.14% revenue improvement**

under the modeled pricing scenario on the historical holdout period.

> This is a model-based scenario estimate, not a guaranteed causal revenue increase.

## 💡 Business Insights

The project demonstrates how a retailer can combine:

**Demand Forecasting → Price Sensitivity → Pricing Optimization**

to support data-driven revenue decisions.

## ⚠️ Limitations

- Pricing relationships are estimated from observational data rather than randomized experiments.
- Unobserved factors may influence customer demand.
- The project optimizes expected revenue rather than profit because product-level cost/margin data is unavailable.
- Real-world deployment would also require competitor pricing, inventory constraints, operational constraints, and customer behavior monitoring.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Statsmodels
- Jupyter Notebook

## 📁 Project Structure

```text
retail-demand-dynamic-pricing/
│
├── README.md
├── retail_demand_forecasting_dynamic_pricing.ipynb
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── images/
    ├── demand_forecast.png
    ├── price_elasticity.png
    └── pricing_optimization.png
