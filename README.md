# Retail Demand Forecasting & Dynamic Pricing Optimization

An end-to-end data science project combining demand forecasting, controlled price-elasticity analysis, and constrained pricing optimization, with an interactive Streamlit demonstration.

## Working Demo

The Streamlit app turns the notebook's finalized modeling pipeline into an interactive working demonstration:

**Meal + Fulfilment Center + Price + Promotions → Demand Forecast → Constrained Price Recommendation → What-if Revenue Simulation**

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The required model, pricing parameters, metadata, and compact demo reference data are included in the repository. You do **not** need the full raw dataset just to run the demo.

## Project Notebook

`retail_demand_dynamic_pricing.ipynb` contains the full analysis and model-development workflow, including:

- Data validation and exploratory analysis
- Lag and rolling demand features
- Time-based train/validation/test design
- XGBoost demand forecasting
- Naive and Ridge baselines
- Controlled log-log price elasticity
- Historical price constraints
- ±20% business price-change constraint
- Holdout pricing scenario evaluation

## Key Results

| Metric | Result |
|---|---:|
| XGBoost MAE | 77.4 orders |
| Naive baseline MAE | 99.1 orders |
| Ridge baseline MAE | 113.3 orders |
| XGBoost improvement vs naive | 21.9% |
| XGBoost improvement vs Ridge | 31.7% |
| Notebook pricing scenario improvement | 10.14% |

The pricing percentage is a **model-based holdout scenario estimate**, not a guaranteed causal revenue increase.

## Repository Structure

```text
retail-demand-dynamic-pricing/
├── app.py
├── retail_demand_dynamic_pricing.ipynb
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   ├── demo_reference.csv
│   └── README.md
│
├── models/
│   ├── demand_xgb.json
│   ├── pricing_parameters.csv
│   ├── metadata.json
│   └── qa_metrics.json
│
└── images/
```

## Pricing Methodology

The interactive application uses controlled category-level price elasticity exported from the finalized notebook. The pricing scenario combines that elasticity with forecasted demand. The theoretical category optimum is bounded by training-period historical prices, then the interactive/holdout stage applies the finalized notebook's maximum ±20% change from the current observed price.

## Limitations

Pricing estimates come from observational historical data rather than a randomized pricing experiment. Unobserved factors may still affect demand. The project optimizes expected revenue rather than profit because product-level costs and margins are unavailable.

## Technology

Python · Pandas · NumPy · XGBoost · Plotly · Streamlit · Statsmodels · Scikit-learn · Jupyter

## Demo Behavior

The top result cards report the model forecast and the final constrained recommendation. The lower **What-if Price Simulation** is an interactive scenario explorer and may show a different value when the slider is moved. This distinction is intentional: recommendation and what-if analysis are separate outputs.

## Interactive Demo

The application separates the **recommended-price scenario** (top cards) from the **what-if price simulation** (slider section). The default slider value is the optimizer recommendation; moving it lets the presenter test alternative feasible prices without changing the underlying recommendation.
