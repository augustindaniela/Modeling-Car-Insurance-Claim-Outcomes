# Modeling Car Insurance Claim Outcomes

Finding the **single feature** with the best predictive power for whether a customer will file a car insurance claim — designed for an insurer with no ML infrastructure that wants to start with a simple model in production.

![car](car.jpg)

## Context

*On the Road* car insurance wants to predict claims, but lacks the infrastructure to maintain complex models. The ask: find one single variable that, on its own, already has decent predictive power — a "minimum viable model."

## Dataset

- `car_insurance.csv`: 16 customer variables (age, gender, driving experience, education, income, credit score, vehicle ownership, vehicle year, children, annual mileage, traffic violations, DUI history, past accidents) and the target variable `outcome` (whether a claim was filed)

## Methodology

1. Handled missing values in `credit_score` and `annual_mileage` (filled with the mean)
2. Correlation heatmap across all numeric variables
3. **Univariate logistic regression** (`statsmodels.logit`) trained separately for **each of the 15 features**, computing each individual model's accuracy via its confusion matrix
4. Visual comparison (bar chart) of accuracy across all models

## Key Findings

- **Best single feature: `driving_experience`**, with **78% accuracy**
- This clearly outperforms every other variable tested individually, suggesting the driver's years of driving experience is, on its own, a strong risk proxy
- The final result was summarized in a simple table: `driving_experience` → 0.78

## Tech Stack

`pandas` · `numpy` · `statsmodels` (logistic regression) · `seaborn` · `matplotlib`
