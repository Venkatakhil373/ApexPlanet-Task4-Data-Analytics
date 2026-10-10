"""
ApexPlanet Data Analytics Internship — Task 4
Data Storytelling & Statistical Validation

Run:
    pip install pandas openpyxl matplotlib scipy
    python analysis.py

Keep the Excel dataset in the same folder as this script, or update DATA_FILE below.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_FILE = Path("ApexPlanet_DataAnalytics_Dataset(3).xlsx")
OUTPUT_DIR = Path("analysis_outputs")
ALPHA = 0.05


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE.resolve()}\n"
            "Place the Excel dataset beside analysis.py or update DATA_FILE."
        )

    # Load and prepare data
    df = pd.read_excel(DATA_FILE)
    required = {"Order_Date", "City", "Category", "Total_Sales"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Total_Sales"] = pd.to_numeric(df["Total_Sales"], errors="coerce")
    df = df.dropna(subset=["Total_Sales", "Category", "City"])

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Overall summary
    print("\\n=== DATASET SUMMARY ===")
    print(f"Rows analyzed: {len(df):,}")
    print(f"Total sales: {df['Total_Sales'].sum():,.2f}")
    print(f"Average transaction sales: {df['Total_Sales'].mean():,.2f}")
    print(f"Median transaction sales: {df['Total_Sales'].median():,.2f}")
    print(f"Cities: {df['City'].nunique()}")
    print(f"Categories: {df['Category'].nunique()}")
    print(f"Missing values in original workbook: {pd.read_excel(DATA_FILE).isna().sum().sum()}")

    # Category analysis
    category_summary = (
        df.groupby("Category")["Total_Sales"]
        .agg(Order_Count="count", Total_Sales="sum", Average_Sales="mean")
        .sort_values("Total_Sales", ascending=False)
    )
    print("\\n=== SALES BY CATEGORY ===")
    print(category_summary.to_string(float_format=lambda x: f"{x:,.2f}"))
    category_summary.to_csv(OUTPUT_DIR / "category_sales_summary.csv")

    ax = category_summary["Total_Sales"].sort_values().plot(kind="barh", figsize=(9, 5))
    ax.set_title("Total Sales by Product Category")
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Category")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_by_category.png", dpi=160)
    plt.close()

    # City analysis
    city_summary = df.groupby("City")["Total_Sales"].agg(["count", "sum", "mean"])
    city_summary = city_summary.sort_values("sum", ascending=False)
    print("\\n=== SALES BY CITY ===")
    print(city_summary.to_string(float_format=lambda x: f"{x:,.2f}"))
    city_summary.to_csv(OUTPUT_DIR / "city_sales_summary.csv")

    ax = city_summary["sum"].sort_values().plot(kind="barh", figsize=(9, 5))
    ax.set_title("Total Sales by City")
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("City")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_by_city.png", dpi=160)
    plt.close()

    # Monthly analysis
    dated = df.dropna(subset=["Order_Date"]).copy()
    dated["Month"] = dated["Order_Date"].dt.to_period("M").astype(str)
    monthly_summary = dated.groupby("Month")["Total_Sales"].agg(["count", "sum", "mean"])
    monthly_summary = monthly_summary.sort_index()
    print("\\n=== MONTHLY SALES ===")
    print(monthly_summary.to_string(float_format=lambda x: f"{x:,.2f}"))
    monthly_summary.to_csv(OUTPUT_DIR / "monthly_sales_summary.csv")

    ax = monthly_summary["sum"].plot(kind="line", marker="o", figsize=(10, 5))
    ax.set_title("Monthly Total Sales")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_sales_trend.png", dpi=160)
    plt.close()

    # One-way ANOVA: compare mean transaction sales across categories
    groups = [
        group["Total_Sales"].dropna().to_numpy()
        for _, group in df.groupby("Category")
        if group["Total_Sales"].notna().sum() > 1
    ]
    print("\\n=== ONE-WAY ANOVA: TOTAL SALES BY CATEGORY ===")
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significance level (alpha): {ALPHA}")
        if p_value < ALPHA:
            print("Decision: Reject H0.")
            print("Interpretation: Evidence suggests at least one category mean differs.")
        else:
            print("Decision: Fail to reject H0.")
            print("Interpretation: Insufficient evidence that category mean sales differ.")
    else:
        print("ANOVA could not be run: at least two categories with multiple observations are required.")

    print(f"\\nAnalysis complete. Tables and charts saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
