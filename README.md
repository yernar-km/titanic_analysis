📊 Titanic Dataset Analysis

Python Data Cleaning • Descriptive Statistics • Visualization

This project provides a full exploratory data analysis (EDA) workflow on the Titanic dataset, including data preprocessing, descriptive statistics generation, and visualization of key survival patterns.
All steps are implemented in Python using Pandas, NumPy, Matplotlib, and Seaborn.

🚀 Features
✅ 1. Data Loading & Preparation

Loads the Titanic dataset directly from seaborn.

Renames several columns for better readability.

Fills missing values:

Age → median age per passenger class (Pclass)

Embarked → most frequent category (mode)

Log-transform of the Fare variable (LogFare).

Converts categorical features:

Sex: male → 0, female → 1

✅ 2. Descriptive Statistics Table

Automatically generates Table 1 with:

Age: mean (SD)

Fare: median (IQR)

Sex distribution: n (%)

Pclass distribution: n (%)

Overall survival rate (%)

All results are printed to the console.

✅ 3. Visualizations (Figures 1–3)
Figure 1 — Survival by Sex

Stacked bar chart showing counts (survived/died)

Bar chart showing survival percentages for men vs women
Saved as: figure1_survival_by_sex.png

Figure 2 — Survival by Passenger Class

Stacked bar chart visualizing outcomes across Pclass
Saved as: figure2_survival_by_pclass.png

Figure 3 — Age Distribution

Left: histogram by survival status

Right: KDE distribution plot
Saved as: figure3_age_distribution.png

📁 Project Structure
