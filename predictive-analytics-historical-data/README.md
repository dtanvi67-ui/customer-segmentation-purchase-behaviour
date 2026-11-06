# Predictive Analytics Using Historical Data

A practical forecasting project that uses historical monthly sales data to predict the next 12 months. It demonstrates data cleaning, time-based validation, regression, model evaluation, and forecast visualization.

> **Dataset note:** The included monthly sales data is synthetic and generated for learning. It is not real company or customer data.

## Tools
- Python 3.10+
- Pandas, NumPy
- Scikit-learn (Random Forest Regression)
- Matplotlib
- Power BI Desktop (optional dashboard)

## Project structure
```text
predictive-analytics-historical-data/
├── data/
│   └── monthly_sales.csv
├── src/
│   └── forecast_sales.py
├── outputs/                  # created when the script runs
├── requirements.txt
├── .gitignore
└── README.md
```

## Run in VS Code (Windows)
1. Install Python 3.10 or later.
2. Open this project folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate a virtual environment:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Run:
   ```bash
   python src/forecast_sales.py
   ```
   If `py` is not recognized, use `python`. If PowerShell blocks activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` followed by `.venv\Scripts\python.exe src/forecast_sales.py`.

## Methodology
1. Load monthly historical sales and parse dates.
2. Sort chronologically and handle missing sales values.
3. Create calendar, seasonal, lag, and rolling-average features.
4. Train a Random Forest regression model using earlier months.
5. Evaluate on the final 12 historical months using a rolling, recursive forecast (so test-period actual values are not used as future lag inputs).
6. Report MAE, RMSE, and MAPE.
7. Retrain on all available history and forecast the next 12 months.

The forecast is an estimate, not a guarantee. Results depend on data quality and whether historical patterns continue.

## Output files
After running the script, the `outputs/` folder contains:
- `model_evaluation.txt` — MAE, RMSE, and MAPE
- `test_predictions.csv` — actual vs predicted values for the holdout period
- `future_forecast.csv` — next 12 monthly forecasts
- `historical_and_forecast.png` — chart of history and forecast

## Power BI dashboard
1. Run the Python script first.
2. Open Power BI Desktop.
3. Import `outputs/future_forecast.csv` and `outputs/test_predictions.csv` using **Get data > Text/CSV**.
4. Create a line chart with Date on the X-axis and ForecastSales on the Y-axis.
5. Create a line chart comparing ActualSales and PredictedSales for the test period.
6. Add KPI cards for MAE, RMSE, and MAPE using values from `outputs/model_evaluation.txt` (or enter them into a small manual KPI table).
7. Add a date slicer if desired.
8. Save as `Predictive_Analytics_Dashboard.pbix`.

## Limitations
- Sample data is synthetic.
- Only one model and a simple feature set are used as a beginner-friendly baseline.
- External factors such as promotions, holidays, stock availability, and economic changes are not included.
- Do not use the sample forecast for real financial or business decisions.

## Authoring / GitHub
This project is stored in the `predictive-analytics-historical-data/` folder of the author's project repository.
