# Customer Segmentation and Purchase Behaviour Analysis

A beginner-friendly machine learning project that groups customers by purchasing behaviour using **K-Means clustering** and visualizes the results for business insights.

## Objective
Identify customer groups using annual income, spending score, annual orders, and monthly website visits. Explore category preferences and use the results to plan more relevant customer engagement.

## Technology stack
- Python 3.10+
- Pandas and NumPy
- Scikit-learn (StandardScaler and K-Means)
- Matplotlib and Seaborn
- Power BI Desktop (optional dashboard)

## Project structure
```text
customer-segmentation-purchase-behaviour/
├── data/
│   └── customers.csv       # generated automatically on first run
├── src/
│   └── customer_segmentation.py
├── outputs/                # generated charts and CSV reports
├── requirements.txt
├── .gitignore
└── README.md
```

## Run in VS Code (Windows)
1. Install Python 3.10 or later.
2. Clone or download this repository and open the folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate a virtual environment:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
6. Run the project:
   ```bash
   python src/customer_segmentation.py
   ```
7. The script creates a 500-row synthetic customer dataset in `data/customers.csv` if it does not already exist. It then saves the segmented data, cluster profile, and charts in `outputs/`.

If the `py` command is not recognized, use `python`. If PowerShell blocks virtual-environment activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and then `.venv\Scripts\python.exe src/customer_segmentation.py`.

## Dataset fields
| Column | Description |
|---|---|
| CustomerID | Unique sample customer identifier |
| Age | Customer age |
| AnnualIncome | Sample annual income |
| SpendingScore | Sample spending measure |
| AnnualOrders | Number of orders in a year |
| WebsiteVisitsPerMonth | Monthly website visits |
| PreferredCategory | Preferred product category |
| City | Sample city |

**Dataset note:** The included workflow generates synthetic sample records for learning and demonstration. They are not real customer records.

## Methodology
1. Load the customer dataset.
2. Fill missing numeric values with the median.
3. Standardize the selected numeric features.
4. Apply K-Means clustering with four groups.
5. Calculate a silhouette score and inspect cluster profiles.
6. Export customer labels, summary tables, and visualizations.

Cluster IDs (0–3) are arbitrary. Assign business-friendly names only after reviewing the cluster profile.

## Power BI dashboard
After running the Python script:
1. Open Power BI Desktop.
2. Select **Get data > Text/CSV** and import `outputs/segmented_customers.csv`.
3. Add cards for total customers and average spending score.
4. Add a clustered column chart with `Cluster` as Axis and count of `CustomerID` as Values.
5. Add a scatter chart with `AnnualIncome` on X, `SpendingScore` on Y, and `Cluster` as Legend.
6. Add a bar chart with `PreferredCategory` as Axis, count of `CustomerID` as Values, and `Cluster` as Legend.
7. Add slicers for `City` and `PreferredCategory`.
8. Save the report as `Customer_Segmentation_Dashboard.pbix`.

## Business questions
- Which cluster has the highest average spending score?
- Which group places the most orders?
- Does website engagement differ by cluster?
- Which product categories are popular in each group?
- Which groups may respond to loyalty rewards or product recommendations?

## Limitations
- Synthetic data is used to demonstrate the workflow.
- Four clusters are a starting point; real projects should compare different cluster counts and evaluate business usefulness.
- Clustering reveals patterns, not customer motivation or causation.
