# Real-World-Dataset-Analysis-Project
A Python project analyzing the Titanic dataset: cleans missing data, engineers features, creates 9 visualizations, runs statistical tests and a logistic regression, and reports key insights.
Real-World Dataset Analysis Project

An end-to-end analysis of the Titanic passenger dataset: data loading, exploratory data analysis (EDA), cleaning and preprocessing, visualization, statistical testing, a baseline model, and written interpretation of the findings.

Objective

Select a real-world dataset, perform exploratory data analysis, clean and preprocess the data, and present insights with visualizations and interpretations.

Dataset
Item	Detail
Name	Titanic passenger records
Size	891 passengers, 15 columns
Source	Loaded automatically from seaborn's public data repository (sns.load_dataset("titanic"))
Target question	Which factors influenced whether a passenger survived?

Key columns: survived, pclass, sex, age, sibsp, parch, fare, embarked.

Project Structure
real_world_dataset_project/
├── titanic_analysis.py   # main analysis script
├── requirements.txt      # Python dependencies
├── README.md             # this file
└── output/               # created when you run the script
    ├── 01_survival_overview.png
    ├── 02_survival_by_category.png
    ├── 03_age_distribution.png
    ├── 04_survival_by_age_group.png
    ├── 05_fare_analysis.png
    ├── 06_family_size.png
    ├── 07_class_sex_interaction.png
    ├── 08_correlation_heatmap.png
    ├── 09_model_coefficients.png
    ├── titanic_cleaned.csv
    └── analysis_report.txt
Setup

Python 3.9 or later is recommended.

bash
# 1. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
How to Run
bash
python titanic_analysis.py

Options:

Option	Description	Default
--output	Folder where figures, cleaned data and the report are saved	output
--csv	Path to a local Titanic-format CSV (seaborn column names) instead of downloading	none

Example:

bash
python titanic_analysis.py --output results

An internet connection is needed on the first run only if you do not pass --csv.

Methodology
1. Data overview

Shape, data types, first rows, missing values, duplicates and summary statistics are inspected.

2. Cleaning and preprocessing
Step	Action	Reason
Missing age (about 20%)	Filled with the median age of the passenger's class and sex	Preserves group differences better than one global median
Missing embarked	Filled with the most common port	Only 2 values missing
deck column	Dropped	About 77% missing
Redundant columns	embark_town, alive, class, who, adult_male dropped	Duplicate information in other columns
Duplicate rows	Reported, not removed	No passenger ID exists, so identical rows are usually different people
Fare outliers	Flagged using the IQR rule, kept	High fares are genuine first-class tickets
3. Feature engineering

family_size, is_alone, age_group (Child, Teen, Young Adult, Adult, Senior) and fare_log (log-transformed fare to reduce skew).

4. Exploratory analysis and visualization

Nine figures cover survival counts, survival by sex, class, embarkation port, age, fare, family size, the class and sex interaction, and a correlation heatmap.

5. Statistical testing

Chi-square tests (sex vs survival, class vs survival) and Welch t-tests (age and fare, survivors vs non-survivors).

6. Baseline model

A logistic regression with standardized features, used mainly to interpret which variables matter rather than to maximize accuracy.

Key Findings

Figures below come from a run of the script; your values should match.

Overall survival rate: 38.4%.
Sex was the strongest factor: 74% of women survived versus 19% of men.
Class mattered: 63% survival in first class, 47% in second, 24% in third.
Class and sex interact: about 97% of first-class women survived versus 14% of third-class men.
Children (under 13) survived at about 58%, above the overall average.
Travelling alone was riskier (30% survival) than travelling with family (51%), but families of five or more fared poorly.
Fare is positively linked to survival, but largely because it reflects passenger class.
Statistical tests: sex, class and fare differences are significant (p < 0.05). The age difference between survivors and non-survivors was not significant at the 5% level (p ≈ 0.08).
Baseline model: logistic regression reached about 82% test accuracy, with sex and class as the largest coefficients.
Limitations
The data describes one historical event, so findings do not generalize to other situations.
About 20% of ages were imputed, which may blur age effects.
The results show association, not causation.
The logistic regression is a simple baseline; stronger models could do better.
Possible Extensions
Extract passenger titles (Mr, Mrs, Miss, Master) from names as a feature.
Compare models such as Random Forest and Gradient Boosting using cross-validation.
Build an interactive dashboard with Plotly or Streamlit.
Repeat the workflow on another real-world dataset (for example, housing prices or hospital data).
Tools Used

Python, pandas, NumPy, Matplotlib, seaborn, SciPy, scikit-learn.
