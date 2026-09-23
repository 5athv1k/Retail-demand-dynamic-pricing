# Streamlit Demo

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What is included

- `models/demand_xgb.json`: saved XGBoost demand model
- `models/pricing_parameters.csv`: controlled elasticity and category pricing parameters exported from the finalized notebook
- `models/metadata.json`: model provenance and benchmark metrics
- `data/demo_reference.csv`: latest historical features for each center-meal pair

## Design safeguards

- Current price is preserved as the scenario input; an out-of-range warning is shown when it falls outside the training-period category history.
- The final recommendation enforces the finalized notebook's ±20% change rule around the current price; the theoretical category optimum is training-period historically bounded.
- The **Predict & Optimize** button commits the scenario; changing inputs does not silently replace the last submitted scenario.
- The UI distinguishes model-based scenario estimates from causal business outcomes.

## Reproducibility

The full analytical workflow is in `retail_demand_dynamic_pricing.ipynb`. The deployed demo uses the saved artifacts so a user can run the application without carrying the full raw dataset into the repository.

## Output interpretation

- **Top cards:** forecast and recommended-price scenario.
- **What-if section:** user-selected alternative price scenario.
- **Recommendation:** always derived from the constrained optimizer.
