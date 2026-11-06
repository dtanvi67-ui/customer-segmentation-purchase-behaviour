"""
Predictive Analytics Using Historical Data
Monthly sales forecasting with Random Forest regression.

Run from this project folder:
    python src/forecast_sales.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "monthly_sales.csv"
OUTPUT_DIR = ROOT / "outputs"
HORIZON = 12
FEATURE_COLUMNS = [
    "time_index", "year", "month", "month_sin", "month_cos",
    "lag_1", "lag_12", "rolling_mean_3"
]


def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    data = data.sort_values("Date").drop_duplicates(subset="Date")
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    data["Sales"] = data["Sales"].interpolate(limit_direction="both")
    if data["Sales"].isna().any():
        raise ValueError("Sales column contains missing values that could not be filled.")
    if len(data) < 30:
        raise ValueError("Please provide at least 30 monthly observations.")
    return data.reset_index(drop=True)


def make_features(history, date, time_index):
    """Create predictors using only values available before the forecast date."""
    month = date.month
    return {
        "time_index": time_index,
        "year": date.year,
        "month": month,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "lag_1": history[-1],
        "lag_12": history[-12] if len(history) >= 12 else history[-1],
        "rolling_mean_3": float(np.mean(history[-3:])),
    }


def build_training_frame(data):
    values = data["Sales"].astype(float).tolist()
    rows = []
    for i in range(12, len(data)):
        row = make_features(values[:i], data.loc[i, "Date"], i)
        row["Sales"] = values[i]
        rows.append(row)
    return pd.DataFrame(rows)


def recursive_forecast(model, history, start_date, start_index, periods):
    history_values = list(map(float, history))
    predictions = []
    dates = pd.date_range(start=start_date, periods=periods, freq="MS")
    for step, date in enumerate(dates):
        features = make_features(history_values, date, start_index + step)
        x = pd.DataFrame([features])[FEATURE_COLUMNS]
        prediction = max(0.0, float(model.predict(x)[0]))
        predictions.append(prediction)
        history_values.append(prediction)
    return dates, predictions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    if len(data) <= HORIZON + 12:
        raise ValueError("Need more history to create a 12-month holdout and training set.")

    split = len(data) - HORIZON
    train_data = data.iloc[:split].copy()
    test_data = data.iloc[split:].copy()

    train_frame = build_training_frame(train_data)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["Sales"])

    test_dates, test_predictions = recursive_forecast(
        model=model,
        history=train_data["Sales"].tolist(),
        start_date=test_data.iloc[0]["Date"],
        start_index=split,
        periods=len(test_data)
    )
    actual = test_data["Sales"].to_numpy(dtype=float)
    predicted = np.asarray(test_predictions, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    nonzero = np.abs(actual) > 1e-9
    mape = (np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
            if nonzero.any() else float("nan"))

    test_output = pd.DataFrame({
        "Date": test_dates,
        "ActualSales": actual,
        "PredictedSales": predicted
    })
    test_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Fit a final model using all historical observations, then forecast 12 future months.
    full_frame = build_training_frame(data)
    final_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(full_frame[FEATURE_COLUMNS], full_frame["Sales"])
    last_date = data["Date"].max()
    future_start = last_date + pd.offsets.MonthBegin(1)
    future_dates, future_values = recursive_forecast(
        model=final_model,
        history=data["Sales"].tolist(),
        start_date=future_start,
        start_index=len(data),
        periods=HORIZON
    )
    future = pd.DataFrame({
        "Date": future_dates,
        "ForecastSales": np.round(future_values, 2)
    })
    future.to_csv(OUTPUT_DIR / "future_forecast.csv", index=False)

    metrics_text = (
        "Random Forest Regression - Holdout Evaluation\n"
        f"Holdout months: {len(test_data)}\n"
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"MAPE: {mape:.2f}%\n"
        "\nMetrics are calculated on the final 12 historical months.\n"
        "The test forecast is recursive; actual holdout values are not fed back as lag inputs.\n"
    )
    (OUTPUT_DIR / "model_evaluation.txt").write_text(metrics_text, encoding="utf-8")

    plt.figure(figsize=(12, 6))
    plt.plot(data["Date"], data["Sales"], label="Historical sales")
    plt.plot(future["Date"], future["ForecastSales"], marker="o",
             linestyle="--", label="12-month forecast")
    plt.axvline(last_date, linestyle=":", label="Forecast starts")
    plt.title("Monthly Sales: Historical Trend and 12-Month Forecast")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "historical_and_forecast.png", dpi=160)
    plt.close()

    print(metrics_text)
    print(f"Files saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
