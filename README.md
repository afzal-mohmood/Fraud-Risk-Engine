# Fraud Detection & Risk Scoring Pipeline

## Project Overview

This project builds an end-to-end machine learning pipeline to detect fraudulent payment transactions. It covers the full data science workflow — from raw data ingestion and cleaning, through feature engineering and model training, to a business-facing Power BI dashboard that makes the results actionable.

This project was built to mirror real-world challenges in payment fraud detection, directly relevant to fintech environments where reducing false declines and catching genuine fraud both matter.

- **Data Source:** [Kaggle — Fraud Detection Transactions Dataset by Samay Ashar](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset)
- **Dataset Size:** 50,000 transactions, 21 features per record
- **Tech Stack:** Python (Pandas, NumPy, scikit-learn), Jupyter Notebook, Power BI Desktop

---

## How the Pipeline Works

The project follows a clear, step-by-step workflow:

### 1. Data Ingestion & Inspection
Load the raw dataset and profile its structure — checking column types, identifying missing values, and understanding the class balance between genuine and fraudulent transactions.

### 2. Data Cleaning & Feature Engineering
- Removed `Transaction_ID` and `User_ID` — these are unique row identifiers with no predictive value; keeping them would cause the model to memorise rather than learn
- Parsed timestamps from mixed string formats into proper datetime objects
- Extracted `Hour_of_Day` (0–23) as a numerical feature — transaction timing is a meaningful fraud signal
- Dropped the original timestamp column after extraction

### 3. Encoding Categorical Variables
Converted text-based columns (`Device_Type`, `Location`, `Transaction_Type`, `Merchant_Category`, `Card_Type`, `Authentication_Method`) into numerical binary flags using one-hot encoding, which is required for scikit-learn models to process them.

### 4. Fixing Data Leakage
The raw dataset included a `Risk_Score` column — a pre-calculated fraud score that would give the model access to information it wouldn't have in a real deployment. This was identified and removed before training. The model was then retrained on genuine behavioural features only, giving a realistic view of performance.

### 5. Model Training
Trained a Random Forest Classifier on an 80/20 train-test split. The `stratify` parameter was used to ensure both splits maintained the same fraud-to-genuine ratio. `class_weight='balanced'` was applied to prevent the model from ignoring the minority fraud class.

### 6. Exporting for Visualisation
The cleaned dataset was exported as a CSV for loading into Power BI to build the business dashboard.

---

## Model Results

The model was evaluated on 10,000 unseen transactions after removing the leaking `Risk_Score` column:

| Class | Precision | Recall | F1-Score | Support |
|:---|:---|:---|:---|:---|
| **0 — Genuine** | 0.85 | 1.00 | 0.92 | 6,787 |
| **1 — Fraud** | 1.00 | 0.62 | 0.76 | 3,213 |
| **Overall Accuracy** | | | **0.88** | 10,000 |
| **Macro Average** | 0.92 | 0.81 | 0.84 | 10,000 |

### What these numbers mean in plain terms

- **Fraud Precision — 100%:** Every transaction the model flagged as fraud was genuinely fraudulent. There were zero false positives — meaning no legitimate customer transactions would have been incorrectly blocked.
- **Fraud Recall — 62%:** The model correctly caught 62% of all actual fraud cases using only behavioural signals (time of day, device type, merchant category, location, card type, authentication method). This is a realistic baseline without any pre-scored risk features.
- **Note:** The 100% precision figure reflects characteristics of this specific dataset. In a live production environment, some false positives would be expected and the threshold would be tuned based on the business's tolerance for blocking genuine transactions vs. missing fraud.

---

## Power BI Dashboard

The cleaned output data was connected to Power BI Desktop to build an operational fraud monitoring dashboard.

![Dashboard Preview](dashboard_preview.png)

### What the dashboard shows

1. **Fraud by Hour of Day** — A line chart showing when fraud peaks across a 24-hour window, useful for scheduling manual review teams
2. **Geographic Fraud Distribution** — A map visual showing which regions show higher fraud concentrations
3. **Fraud by Authentication Method & Merchant Category** — Bar charts breaking down which transaction types and authentication channels are most associated with fraud attempts

---

## Running This Project Yourself

### Step 1 — Install dependencies

```bash
pip install pandas numpy scikit-learn
```

### Step 2 — Get the data

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset), place it in the project root folder, and name it `fraud_dataset.csv`.

### Step 3 — Run the pipeline

```bash
python fraud_risk_pipeline.py
```

This will print the model evaluation report to your terminal and save `cleaned_fraud_visualization.csv` to your project folder for use in the dashboard.

---

## Why This Project Matters

Clean, trusted data is the foundation of any reliable ML system. A fraud model built on leaky or poorly prepared data will perform well in testing but fail in production — a costly mistake in a payments context. This project deliberately identifies and fixes a data leakage issue to demonstrate that awareness.

The goal was not just to build a model, but to build one that could be explained to a business stakeholder and trusted in a real environment.

---

*Built as a data science portfolio project | Tools: Python · scikit-learn · Pandas · Power BI*
