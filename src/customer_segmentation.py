"""
Customer Segmentation and Purchase Behaviour Analysis
Run from the repository root:
    python src/customer_segmentation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA = DATA_DIR / "customers.csv"
OUT = ROOT / "outputs"
FEATURES = [
    "AnnualIncome",
    "SpendingScore",
    "AnnualOrders",
    "WebsiteVisitsPerMonth",
]


def create_sample_dataset(path: Path, rows: int = 500) -> None:
    """Create reproducible synthetic customer records for demonstration."""
    rng = np.random.default_rng(42)
    segment_names = ["budget", "regular", "premium", "occasional"]
    segment_probabilities = [0.30, 0.35, 0.20, 0.15]
    segments = rng.choice(segment_names, size=rows, p=segment_probabilities)

    income = np.zeros(rows)
    spending = np.zeros(rows)
    orders = np.zeros(rows)
    visits = np.zeros(rows)

    distributions = {
        "budget": (35000, 9000, 900, 250, 18, 5, 2.5, 1.0),
        "regular": (65000, 14000, 2400, 450, 35, 8, 5.0, 1.5),
        "premium": (115000, 22000, 5200, 900, 55, 12, 8.0, 2.0),
        "occasional": (52000, 16000, 1300, 400, 8, 3, 1.5, 0.8),
    }
    for i, segment in enumerate(segments):
        values = distributions[segment]
        income[i] = rng.normal(values[0], values[1])
        spending[i] = rng.normal(values[2], values[3])
        orders[i] = rng.normal(values[4], values[5])
        visits[i] = rng.normal(values[6], values[7])

    data = pd.DataFrame({
        "CustomerID": [f"CUST{i:04d}" for i in range(1, rows + 1)],
        "Age": rng.integers(18, 66, rows),
        "AnnualIncome": np.clip(income, 15000, None).round().astype(int),
        "SpendingScore": np.clip(spending, 100, 8000).round().astype(int),
        "AnnualOrders": np.clip(orders, 1, 100).round().astype(int),
        "WebsiteVisitsPerMonth": np.clip(visits, 0, 20).round().astype(int),
        "PreferredCategory": rng.choice(
            ["Fashion", "Electronics", "Home", "Beauty", "Grocery"], rows
        ),
        "City": rng.choice(
            ["Nashik", "Pune", "Mumbai", "Nagpur", "Other"],
            rows,
            p=[0.25, 0.25, 0.25, 0.15, 0.10],
        ),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    print(f"Created sample dataset: {path} ({rows} customers)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        create_sample_dataset(DATA)

    df = pd.read_csv(DATA)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[FEATURES])

    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(scaled_features)

    score = silhouette_score(scaled_features, df["Cluster"])
    print(f"Silhouette score: {score:.3f}")
    print("\nCustomer counts by cluster:")
    print(df["Cluster"].value_counts().sort_index())

    profile = df.groupby("Cluster")[FEATURES].mean().round(1)
    profile["CustomerCount"] = df.groupby("Cluster").size()
    profile.to_csv(OUT / "cluster_profile.csv")
    df.to_csv(OUT / "segmented_customers.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="AnnualIncome", y="SpendingScore",
        hue="Cluster", palette="Set2", s=65
    )
    plt.title("Customer Segments: Income vs Spending")
    plt.tight_layout()
    plt.savefig(OUT / "customer_segments.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.heatmap(profile[FEATURES], annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Average Behaviour by Customer Cluster")
    plt.tight_layout()
    plt.savefig(OUT / "cluster_profile_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x="PreferredCategory", hue="Cluster")
    plt.title("Preferred Product Category by Cluster")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT / "category_preferences.png", dpi=160)
    plt.close()

    print(f"\nOutputs saved in: {OUT}")
    print("Import outputs/segmented_customers.csv into Power BI.")


if __name__ == "__main__":
    main()
