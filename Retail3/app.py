from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

FEATURE_COLS = [
    "checkout_price", "base_price", "discount", "discount_pct",
    "emailer_for_promotion", "homepage_featured", "category_code",
    "cuisine_code", "center_type_code", "op_area", "orders_lag_1",
    "orders_lag_2", "orders_roll_mean_4",
]

BUSINESS_CONSTRAINT = 0.20

st.set_page_config(
    page_title="Retail Demand & Dynamic Pricing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def money(value):
    return f"₹{float(value):,.2f}"


def whole(value):
    return f"{int(round(float(value))):,}"


def p_value_label(value):
    value = float(value)
    if value == 0:
        return "p < 1e-300"
    if value < 0.001:
        return f"p = {value:.2e}"
    return f"p = {value:.4f}"


@st.cache_resource
def load_model():
    path = MODEL_DIR / "demand_xgb.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


@st.cache_data
def load_reference_data():
    path = DATA_DIR / "demo_reference.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference data: {path}")

    data = pd.read_csv(path)

    # The forecasting model requires lag/rolling history. A few sparse
    # center-meal combinations in the source history do not have those
    # features; exclude them from the interactive selector rather than
    # allowing a runtime prediction failure.
    required_reference_cols = [
        "category_code",
        "cuisine_code",
        "center_type_code",
        "op_area",
        "orders_lag_1",
        "orders_lag_2",
        "orders_roll_mean_4",
    ]
    data = data.dropna(subset=required_reference_cols).copy()
    return data


@st.cache_data
def load_pricing_parameters():
    path = MODEL_DIR / "pricing_parameters.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pricing parameters: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_metadata():
    path = MODEL_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_features(row, current_price, base_price, email_promo, homepage_featured):
    discount = float(base_price) - float(current_price)
    discount_pct = discount / float(base_price) if base_price else 0.0

    values = {
        "checkout_price": float(current_price),
        "base_price": float(base_price),
        "discount": discount,
        "discount_pct": discount_pct,
        "emailer_for_promotion": int(email_promo),
        "homepage_featured": int(homepage_featured),
        "category_code": row["category_code"],
        "cuisine_code": row["cuisine_code"],
        "center_type_code": row["center_type_code"],
        "op_area": row["op_area"],
        "orders_lag_1": row["orders_lag_1"],
        "orders_lag_2": row["orders_lag_2"],
        "orders_roll_mean_4": row["orders_roll_mean_4"],
    }
    return pd.DataFrame([[values[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)


def forecast_demand(model, row, price, base_price, email_promo, homepage_featured):
    X = build_features(row, price, base_price, email_promo, homepage_featured)
    log_pred = float(model.predict(X)[0])
    return max(0.0, float(np.expm1(log_pred)))


def business_price_bounds(current_price):
    """Return the ±20% business price-change interval around the current price.

    The notebook's category optimizer already constrains the theoretical category
    optimum to its training-period historical price range. The holdout/demo stage
    then applies the business rule relative to the current observed price. This
    mirrors the finalized notebook backtest and also handles holdout prices that
    fall outside the training-period category range without silently changing the
    user's input.
    """
    current_price = float(current_price)
    if not np.isfinite(current_price) or current_price <= 0:
        raise ValueError("Current price must be a positive finite value.")

    lower = current_price * (1 - BUSINESS_CONSTRAINT)
    upper = current_price * (1 + BUSINESS_CONSTRAINT)

    return lower, upper


def constrained_recommendation(category_params, current_price):
    elasticity = float(category_params["elasticity"])
    historical_min = float(category_params["min_price"])
    historical_max = float(category_params["max_price"])
    theoretical_price = float(category_params["optimal_price"])

    lower, upper = business_price_bounds(current_price)
    recommended = float(np.clip(theoretical_price, lower, upper))

    return recommended, lower, upper, elasticity, historical_min, historical_max, theoretical_price


# ---------------------------
# Load artifacts
# ---------------------------
try:
    model = load_model()
    reference = load_reference_data()
    pricing = load_pricing_parameters()
    metadata = load_metadata()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


# ---------------------------
# Sidebar: scenario inputs
# ---------------------------
st.sidebar.title("Pricing Scenario")
st.sidebar.caption("Inputs are evaluated against the saved project model and controlled elasticity parameters.")

meal_options = (
    reference[["meal_id", "meal_label"]]
    .drop_duplicates()
    .sort_values("meal_id")
)

meal_label = st.sidebar.selectbox(
    "Select meal",
    meal_options["meal_label"].tolist(),
)
selected_meal_id = int(
    meal_options.loc[
        meal_options["meal_label"] == meal_label, "meal_id"
    ].iloc[0]
)

centers_for_meal = reference[reference["meal_id"] == selected_meal_id].sort_values("center_id")
center_id = st.sidebar.selectbox(
    "Select fulfilment center",
    centers_for_meal["center_id"].astype(int).tolist(),
)

scenario_row = reference[
    (reference["meal_id"] == selected_meal_id)
    & (reference["center_id"] == center_id)
].iloc[0]

price_params = pricing[pricing["category"] == scenario_row["category"]].iloc[0]
historical_min = float(price_params["min_price"])
historical_max = float(price_params["max_price"])

# Preserve the actual selected meal-center price as the scenario starting point.
# The final recommendation applies the notebook's ±20% business constraint around
# this current price. A warning is shown when the current price is outside the
# training-period category range rather than silently changing the input.
reference_price = float(scenario_row["checkout_price"])
current_price = st.sidebar.number_input(
    "Current checkout price (₹)",
    min_value=0.01,
    value=reference_price,
    step=0.50,
    help="Starting price for the scenario. The recommendation is constrained to ±20% from this price.",
)

base_price = st.sidebar.number_input(
    "Base price (₹)",
    min_value=0.01,
    value=float(scenario_row["base_price"]),
    step=0.50,
)

email_promo = st.sidebar.checkbox(
    "Email promotion",
    value=bool(scenario_row["emailer_for_promotion"]),
)

homepage_featured = st.sidebar.checkbox(
    "Homepage featured",
    value=bool(scenario_row["homepage_featured"]),
)

run = st.sidebar.button(
    "🔮 Predict & Optimize",
    type="primary",
    use_container_width=True,
)

current_signature = (
    int(selected_meal_id), int(center_id), round(float(current_price), 2),
    round(float(base_price), 2), int(email_promo), int(homepage_featured)
)

if run:
    st.session_state["committed_scenario"] = current_signature

committed = st.session_state.get("committed_scenario")

# ---------------------------
# Main content
# ---------------------------
st.title("📊 Retail Demand Forecasting & Dynamic Pricing Optimization")
st.caption(
    "Interactive demonstration of the finalized demand-forecasting and controlled pricing pipeline."
)

with st.expander("ℹ️ How the demo works", expanded=False):
    st.write(
        "Select a meal and fulfilment center, enter a scenario price and promotion settings, "
        "then click **Predict & Optimize**. The app forecasts demand and evaluates a constrained "
        "pricing scenario using the project's controlled elasticity estimate."
    )

if committed is None:
    st.info("Set your scenario inputs in the sidebar and click **Predict & Optimize** to run the model.")

    perf = metadata["performance"]
    st.subheader("📌 Project Benchmark")
    c1, c2, c3 = st.columns(3)
    c1.metric("XGBoost MAE", f"{perf['notebook_xgboost_mae']:.1f} orders")
    c2.metric("Naive MAE", f"{perf['naive_mae']:.1f} orders")
    c3.metric("Ridge MAE", f"{perf['ridge_mae']:.1f} orders")
    st.stop()

# Read the last submitted values, not live unsaved widget edits.
meal_id_c, center_id_c, current_price_c, base_price_c, email_c, homepage_c = committed
scenario_row = reference[
    (reference["meal_id"] == meal_id_c)
    & (reference["center_id"] == center_id_c)
].iloc[0]
price_params = pricing[pricing["category"] == scenario_row["category"]].iloc[0]

if current_signature != committed:
    st.warning("Inputs have changed. Click **Predict & Optimize** to apply the new scenario.")

if current_price_c < historical_min or current_price_c > historical_max:
    st.warning(
        f"Current price {money(current_price_c)} is outside the training-period "
        f"historical category range ({money(historical_min)}–{money(historical_max)}). "
        "The app will not modify your input; the recommendation is still limited to ±20% "
        "from the current price, matching the notebook's holdout pricing backtest."
    )

if base_price_c < current_price_c:
    st.warning(
        "Current checkout price is above the base price. This is treated as a pricing scenario, "
        "not as a conventional discount."
    )

forecast = forecast_demand(
    model,
    scenario_row,
    current_price_c,
    base_price_c,
    email_c,
    homepage_c,
)

(
    recommended_price,
    lower_price,
    upper_price,
    elasticity,
    historical_min,
    historical_max,
    theoretical_price,
) = constrained_recommendation(price_params, current_price_c)


def scenario_for_price(price):
    ratio = float(price) / float(current_price_c)
    demand = forecast * (ratio ** elasticity)
    revenue = float(price) * demand
    return float(max(0.0, demand)), float(max(0.0, revenue))

recommended_demand, recommended_revenue = scenario_for_price(recommended_price)
current_revenue = float(current_price_c) * forecast
price_change_pct = (recommended_price / current_price_c - 1) * 100
revenue_change_pct = (
    (recommended_revenue / current_revenue - 1) * 100
    if current_revenue else 0.0
)

# ---------------------------
# Top-line result
# ---------------------------
st.subheader("🎯 Demand Forecast & Pricing Recommendation")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Forecasted Demand", f"{whole(forecast)} orders")
with m2:
    st.metric("Recommended Price", money(recommended_price), f"{price_change_pct:+.1f}%")
with m3:
    st.metric("Expected Demand @ Recommended", f"{whole(recommended_demand)} orders")
with m4:
    st.metric(
        "Expected Revenue @ Recommended",
        money(recommended_revenue),
        f"{revenue_change_pct:+.1f}% vs current-price scenario",
    )

left, right = st.columns(2)

with left:
    st.markdown("### Selected Scenario")
    st.write(f"**Meal:** {int(meal_id_c)} — {scenario_row['category']}")
    st.write(f"**Cuisine:** {scenario_row['cuisine']}")
    st.write(f"**Fulfilment center:** {int(center_id_c)}")
    st.write(f"**Latest historical week:** {int(scenario_row['latest_week'])}")
    st.write(f"**Current price:** {money(current_price_c)}")
    st.write(f"**Base price:** {money(base_price_c)}")
    st.write(
        f"**Promotions:** Emailer = {'Yes' if email_c else 'No'}, "
        f"Homepage = {'Yes' if homepage_c else 'No'}"
    )

with right:
    st.markdown("### 🔗 Pricing Logic")
    st.write(f"**Controlled price elasticity:** {elasticity:.3f}")
    st.write(f"**Elasticity significance:** {p_value_label(price_params['p_value'])}")
    st.write(f"**Elasticity observations:** {int(price_params['n_obs']):,}")
    st.write(
        f"**Historical category price range:** "
        f"{money(historical_min)} – {money(historical_max)}"
    )
    st.write("**Business constraint:** maximum ±20% from current price")
    st.write("**Historical price range:** used to bound the theoretical category optimum")
    st.write("**Optimization objective:** maximize expected revenue")
    st.write(f"**Theoretical category optimum:** {money(theoretical_price)}")

# ---------------------------
# Interactive simulation
# ---------------------------
st.markdown("### 💰 What-if Price Simulation")
st.caption(
    f"Move the slider to test alternative prices within {money(lower_price)}–{money(upper_price)}. "
    "The recommended price is calculated first; this section lets you test what-if scenarios "
    "under the same ±20% business constraint used by the notebook backtest."
)

candidate_prices = np.linspace(lower_price, upper_price, 81)
curve_revenue = [scenario_for_price(float(p))[1] for p in candidate_prices]

scenario_price = st.slider(
    "Scenario price (₹)",
    min_value=float(lower_price),
    max_value=float(upper_price),
    value=float(recommended_price),
    step=0.01,
)

scenario_demand, scenario_revenue = scenario_for_price(scenario_price)
scenario_revenue_change = (
    (scenario_revenue / current_revenue - 1) * 100
    if current_revenue else 0.0
)

s1, s2, s3 = st.columns(3)
with s1:
    st.metric("Scenario Demand", f"{whole(scenario_demand)} orders")
with s2:
    st.metric("Scenario Revenue", money(scenario_revenue))
with s3:
    st.metric("Revenue vs Current Price", f"{scenario_revenue_change:+.1f}%")

st.caption(
    f"Current price: {money(current_price_c)}  ·  Recommended price: {money(recommended_price)}  ·  "
    f"Selected what-if price: {money(scenario_price)}"
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=candidate_prices,
        y=curve_revenue,
        mode="lines",
        name="Scenario Revenue",
        line=dict(width=3),
    )
)
fig.add_vline(
    x=recommended_price,
    line_dash="dash",
    annotation_text="Recommended",
    annotation_position="top right",
)
fig.add_vline(
    x=current_price_c,
    line_dash="dot",
    annotation_text="Current",
    annotation_position="bottom right",
)
fig.update_layout(
    title="What-if Revenue Across Feasible Candidate Prices",
    xaxis_title="What-if Price (₹)",
    yaxis_title="Estimated Revenue (₹)",
    height=420,
    margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Recommendation and context
# ---------------------------
if recommended_price < current_price_c:
    direction = "decrease"
elif recommended_price > current_price_c:
    direction = "increase"
else:
    direction = "keep"

if direction == "keep":
    st.success(
        f"Recommendation: keep the price approximately at {money(recommended_price)}. "
        "The constrained optimizer does not identify a different feasible price."
    )
else:
    st.success(
        f"Recommendation: {direction} the price to approximately {money(recommended_price)} "
        f"for this scenario. The model estimates {whole(recommended_demand)} orders and "
        f"{money(recommended_revenue)} in scenario revenue."
    )

st.info(
    "Important: this pricing recommendation is a model-based scenario estimate from "
    "observational data. It is not a guarantee of causal revenue improvement."
)

with st.expander("📌 Model Performance & Project Context"):
    perf = metadata["performance"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Notebook XGBoost MAE", f"{perf['notebook_xgboost_mae']:.1f}")
    c2.metric("Naive MAE", f"{perf['naive_mae']:.1f}")
    c3.metric("Ridge MAE", f"{perf['ridge_mae']:.1f}")
    c4.metric("Notebook Pricing Scenario", f"{perf['notebook_pricing_holdout_revenue_improvement_pct']:.2f}%")

    st.write(
        f"**Model configuration:** XGBoost with {metadata['model_provenance']['validation_selected_trees']} trees, "
        f"selected using validation before the final test holdout."
    )
    st.write(
        f"**Holdout period:** weeks {metadata['model_provenance']['final_test_weeks']}"
    )
    st.write(
        f"**Latest historical week available to the demo:** {metadata['latest_historical_week']}"
    )
    st.caption(metadata["demo_artifact_note"])

st.caption(
    "Demo note: forecasting results and pricing recommendations are scenario estimates derived from the finalized notebook pipeline. "
    "The deployed model uses the notebook's finalized 205-tree configuration and 13-feature schema; the notebook-reported benchmark "
    "metrics remain the primary evaluation figures."
)
