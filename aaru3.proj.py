"""
Real-World Dataset Analysis Project
-----------------------------------
Dataset : Titanic passenger data (891 passengers, real historical records)
Goal    : Exploratory data analysis (EDA), data cleaning and preprocessing,
          visualization, and interpretation of what influenced survival.

Run:
    python titanic_analysis.py
    python titanic_analysis.py --csv my_titanic.csv --output results

All figures, the cleaned dataset and a text report are saved in the output folder.
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")

REPORT_LINES = []


def log(text=""):
    """Print a line and keep it for the saved report."""
    print(text)
    REPORT_LINES.append(str(text))


def section(title):
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def load_data(csv_path=None):
    if csv_path:
        log(f"Loading dataset from file: {csv_path}")
        return pd.read_csv(csv_path)
    log("Loading Titanic dataset from seaborn's public data repository...")
    return sns.load_dataset("titanic")


# ----------------------------------------------------------------------
# 2. INITIAL INSPECTION
# ----------------------------------------------------------------------
def inspect_data(df):
    section("1. DATA OVERVIEW")
    log(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    log("\nColumn types:")
    log(df.dtypes.to_string())
    log("\nFirst 5 rows:")
    log(df.head().to_string())

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    pct = (missing / len(df) * 100).round(1)
    log("\nMissing values:")
    log(pd.DataFrame({"missing": missing, "percent": pct}).to_string())
    log(f"\nDuplicate rows: {df.duplicated().sum()}")
    log("\nSummary statistics (numeric):")
    log(df.describe().round(2).to_string())


# ----------------------------------------------------------------------
# 3. CLEANING AND PREPROCESSING
# ----------------------------------------------------------------------
def clean_data(df):
    section("2. DATA CLEANING AND PREPROCESSING")
    df = df.copy()
    rows_before = len(df)

    # 3.1 Duplicates
    #     This dataset has no passenger ID, so identical-looking rows are usually
    #     different people (e.g. siblings travelling together). They are reported
    #     but NOT removed, as dropping them would delete real passengers.
    dupes = df.duplicated().sum()
    log(f"- Found {dupes} rows identical to another row; kept (no unique ID, likely distinct passengers).")

    # 3.2 Drop columns that are mostly empty or redundant
    #     'deck' is ~77% missing; the others repeat information held elsewhere.
    redundant = ["deck", "embark_town", "alive", "class", "who", "adult_male"]
    redundant = [c for c in redundant if c in df.columns]
    df = df.drop(columns=redundant)
    log(f"- Dropped mostly-empty/redundant columns: {redundant}")

    # 3.3 Impute missing age with the median age of the passenger's class and sex
    age_missing = df["age"].isnull().sum()
    df["age"] = df.groupby(["pclass", "sex"])["age"].transform(
        lambda s: s.fillna(s.median())
    )
    df["age"] = df["age"].fillna(df["age"].median())
    log(f"- Filled {age_missing} missing ages with the median of each class/sex group.")

    # 3.4 Impute missing embarkation port with the most common value
    if df["embarked"].isnull().any():
        emb_missing = df["embarked"].isnull().sum()
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
        log(f"- Filled {emb_missing} missing 'embarked' values with the mode.")

    # 3.5 Feature engineering
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
    )
    df["fare_log"] = np.log1p(df["fare"])
    log("- Created features: family_size, is_alone, age_group, fare_log.")

    # 3.6 Outlier check on fare (IQR rule). Flagged, not removed, because
    #     very high fares are genuine first-class tickets, not errors.
    q1, q3 = df["fare"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    n_out = (df["fare"] > upper).sum()
    df["fare_outlier"] = (df["fare"] > upper).astype(int)
    log(f"- Fare outliers (IQR rule, > {upper:.2f}): {n_out} flagged, kept (valid data).")

    log(f"\nRows: {rows_before} -> {len(df)} | Remaining missing values: {df.isnull().sum().sum()}")
    return df


# ----------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS AND VISUALIZATION
# ----------------------------------------------------------------------
def save_fig(name, outdir):
    path = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"  saved: {path}")


def run_eda(df, outdir):
    section("3. EXPLORATORY DATA ANALYSIS")
    overall = df["survived"].mean() * 100
    log(f"Overall survival rate: {overall:.1f}%")

    # Figure 1: Survival overview
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df["survived"].value_counts().rename({0: "Did not survive", 1: "Survived"}).plot(
        kind="bar", ax=ax[0], color=["#c0392b", "#27ae60"], rot=0
    )
    ax[0].set_title("Survival Count")
    ax[0].set_ylabel("Passengers")
    df["survived"].value_counts().rename({0: "Did not survive", 1: "Survived"}).plot(
        kind="pie", ax=ax[1], autopct="%1.1f%%", colors=["#c0392b", "#27ae60"]
    )
    ax[1].set_ylabel("")
    ax[1].set_title("Survival Share")
    save_fig("01_survival_overview.png", outdir)

    # Figure 2: Survival by sex, class, embarkation
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col in zip(ax, ["sex", "pclass", "embarked"]):
        sns.barplot(data=df, x=col, y="survived", ax=a, errorbar=None)
        a.set_title(f"Survival Rate by {col.title()}")
        a.set_ylabel("Survival rate")
        a.set_ylim(0, 1)
    save_fig("02_survival_by_category.png", outdir)

    for col in ["sex", "pclass", "embarked"]:
        rates = (df.groupby(col)["survived"].mean() * 100).round(1)
        log(f"Survival rate (%) by {col}: {rates.to_dict()}")

    # Figure 3: Age distribution, overall and by survival
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["age"], bins=30, kde=True, ax=ax[0], color="steelblue")
    ax[0].set_title("Age Distribution")
    sns.kdeplot(data=df, x="age", hue="survived", fill=True, common_norm=False, ax=ax[1])
    ax[1].set_title("Age Distribution by Survival")
    save_fig("03_age_distribution.png", outdir)

    # Figure 4: Survival by age group
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="age_group", y="survived", errorbar=None, ax=ax)
    ax.set_title("Survival Rate by Age Group")
    ax.set_ylabel("Survival rate")
    ax.set_ylim(0, 1)
    save_fig("04_survival_by_age_group.png", outdir)

    # Figure 5: Fare analysis
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="pclass", y="fare", ax=ax[0])
    ax[0].set_title("Fare by Passenger Class (outliers visible)")
    sns.boxplot(data=df, x="survived", y="fare_log", ax=ax[1])
    ax[1].set_title("Log Fare by Survival")
    save_fig("05_fare_analysis.png", outdir)

    # Figure 6: Family size
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x="family_size", ax=ax[0])
    ax[0].set_title("Family Size Counts")
    sns.barplot(data=df, x="family_size", y="survived", errorbar=None, ax=ax[1])
    ax[1].set_title("Survival Rate by Family Size")
    ax[1].set_ylim(0, 1)
    save_fig("06_family_size.png", outdir)

    # Figure 7: Class x Sex interaction
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", errorbar=None, ax=ax)
    ax.set_title("Survival Rate by Class and Sex")
    ax.set_ylim(0, 1)
    save_fig("07_class_sex_interaction.png", outdir)

    # Figure 8: Correlation heatmap
    num = df[["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size", "is_alone"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    save_fig("08_correlation_heatmap.png", outdir)

    corr = num.corr()["survived"].drop("survived").sort_values(key=abs, ascending=False)
    log("\nCorrelation with survival (strongest first):")
    log(corr.round(3).to_string())


# ----------------------------------------------------------------------
# 5. STATISTICAL TESTS
# ----------------------------------------------------------------------
def run_stats(df):
    section("4. STATISTICAL TESTING")

    # Chi-square: sex vs survival
    ct = pd.crosstab(df["sex"], df["survived"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    log(f"Chi-square (sex vs survival): chi2 = {chi2:.2f}, p = {p:.3g}")

    # Chi-square: class vs survival
    ct = pd.crosstab(df["pclass"], df["survived"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    log(f"Chi-square (class vs survival): chi2 = {chi2:.2f}, p = {p:.3g}")

    # t-test: age of survivors vs non-survivors
    a = df.loc[df["survived"] == 1, "age"]
    b = df.loc[df["survived"] == 0, "age"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    log(f"Welch t-test (age, survivors vs non-survivors): t = {t:.2f}, p = {p:.3g}")

    # t-test: fare
    a = df.loc[df["survived"] == 1, "fare"]
    b = df.loc[df["survived"] == 0, "fare"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    log(f"Welch t-test (fare, survivors vs non-survivors): t = {t:.2f}, p = {p:.3g}")
    log("A p-value below 0.05 indicates a statistically significant difference.")


# ----------------------------------------------------------------------
# 6. SIMPLE PREDICTIVE MODEL (for interpretation)
# ----------------------------------------------------------------------
def run_model(df, outdir):
    section("5. BASELINE MODEL (Logistic Regression)")
    features = ["pclass", "sex", "age", "family_size", "is_alone", "fare_log", "embarked"]
    X = pd.get_dummies(df[features], columns=["sex", "embarked"], drop_first=True)
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)

    log(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
    log("\nClassification report:")
    log(classification_report(y_test, pred, target_names=["Died", "Survived"]))

    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
    log("Standardised coefficients (negative = lowers survival odds):")
    log(coefs.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    coefs.plot(kind="barh", color=["#c0392b" if v < 0 else "#27ae60" for v in coefs], ax=ax)
    ax.set_title("Logistic Regression Coefficients")
    save_fig("09_model_coefficients.png", outdir)


# ----------------------------------------------------------------------
# 7. INSIGHTS
# ----------------------------------------------------------------------
def write_insights(df):
    section("6. KEY INSIGHTS AND INTERPRETATION")
    female = df.loc[df["sex"] == "female", "survived"].mean() * 100
    male = df.loc[df["sex"] == "male", "survived"].mean() * 100
    first = df.loc[df["pclass"] == 1, "survived"].mean() * 100
    third = df.loc[df["pclass"] == 3, "survived"].mean() * 100
    child = df.loc[df["age_group"] == "Child", "survived"].mean() * 100
    alone = df.loc[df["is_alone"] == 1, "survived"].mean() * 100
    family = df.loc[df["is_alone"] == 0, "survived"].mean() * 100

    insights = [
        f"1. Gender was the strongest factor: {female:.0f}% of women survived versus {male:.0f}% of men, "
        "consistent with the 'women and children first' evacuation policy.",
        f"2. Social class mattered: {first:.0f}% of first-class passengers survived versus {third:.0f}% in third class, "
        "reflecting differences in cabin location and access to lifeboats.",
        f"3. Children had a higher survival rate ({child:.0f}%) than the overall average, "
        "supporting the priority given to children.",
        f"4. Travelling alone was riskier ({alone:.0f}% survival) than travelling with family ({family:.0f}%), "
        "though very large families fared poorly.",
        "5. Higher fares are linked to better survival, but fare is largely a proxy for passenger class.",
        "6. Class and sex interact: women in first and second class had very high survival, "
        "while men in third class had the lowest.",
        "7. Statistical tests confirm that sex and class are strongly associated with survival (p < 0.05).",
        "Limitations: the data covers a single event, some ages were imputed, and association does not prove causation.",
    ]
    for line in insights:
        log(line)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-world dataset analysis (Titanic)")
    parser.add_argument("--csv", help="Optional path to a Titanic-format CSV file")
    parser.add_argument("--output", default="output", help="Folder for results (default: output)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_data(args.csv)
    inspect_data(df)
    df_clean = clean_data(df)
    run_eda(df_clean, args.output)
    run_stats(df_clean)
    run_model(df_clean, args.output)
    write_insights(df_clean)

    clean_path = os.path.join(args.output, "titanic_cleaned.csv")
    df_clean.to_csv(clean_path, index=False)
    report_path = os.path.join(args.output, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES))

    log(f"\nCleaned dataset saved to: {clean_path}")
    log(f"Text report saved to: {report_path}")
    log("Analysis complete.")


if __name__ == "__main__":
    main()
