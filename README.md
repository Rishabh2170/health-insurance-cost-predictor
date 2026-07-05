# Health Insurance Cost Predictor

A linear regression model that estimates annual medical expenditure for health insurance customers based on demographic and lifestyle information. Built for **ACME Insurance Inc.** to support automated, explainable premium pricing.

## Problem Statement

ACME Insurance needs to estimate the annual medical expenditure of new customers using:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region of residence

These estimates are used to set the monthly insurance premium offered to each customer. Because pricing decisions affect real customers, the model needs to be **interpretable** — regulators require a clear explanation for every prediction, not just a number.

## Why Linear Regression?

Linear regression was chosen over more complex models (e.g. random forests, gradient boosting) because:

- Each input feature gets a single, interpretable coefficient
- Predictions can be explained as a transparent sum: `base charge + (coefficient × feature value)` for each feature
- It satisfies the regulatory requirement to justify individual predictions
- It performs competitively on this dataset despite its simplicity

## Dataset

The dataset (`medical.csv`) contains historical records for 1,300+ customers with the following columns:

| Column | Description |
|---|---|
| `age` | Age of the primary beneficiary |
| `sex` | Gender of the customer |
| `bmi` | Body mass index |
| `children` | Number of dependents covered by insurance |
| `smoker` | Smoking status |
| `region` | Residential region in the US |
| `charges` | Actual annual medical expenditure (target variable) |

## Approach

1. **Exploratory Data Analysis (EDA)** — visualized distributions and relationships between features and charges, with particular attention to the effect of smoking status
2. **Feature Engineering** — encoded categorical variables (`sex`, `smoker` as binary; `region` as one-hot)
3. **Model Training** — fit a linear regression model using scikit-learn
4. **Evaluation** — assessed model performance using RMSE (Root Mean Squared Error) on a held-out test set
5. **Interpretation** — examined model coefficients to explain the effect of each feature on predicted charges

## Key Findings

- **Smoking status** is the strongest predictor of medical charges by a wide margin
- **Age** and **BMI** have a positive relationship with charges, and their effect is notably amplified for smokers
- **Number of children** has a smaller, positive effect
- **Sex** and **region** have comparatively minor effects on predicted charges

## Project Structure

```
health-insurance-cost-predictor/
├── l1.ipynb        # Main notebook: EDA, feature engineering, model training & evaluation
├── medical.csv     # Historical dataset used to train the model
└── README.md
```

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter
```

### Running the Notebook

```bash
git clone https://github.com/Rishabh2170/health-insurance-cost-predictor.git
cd health-insurance-cost-predictor
jupyter notebook l1.ipynb
```

## Model Usage Example

```python
import pandas as pd

# Example new customer
new_customer = pd.DataFrame([{
    'age': 35,
    'bmi': 27.5,
    'children': 2,
    'smoker_code': 0,   # 0 = non-smoker, 1 = smoker
    'sex_code': 1,       # 0 = male, 1 = female
    'northeast': 0, 'northwest': 1, 'southeast': 0, 'southwest': 0
}])

predicted_charge = model.predict(new_customer)
print(f"Estimated annual medical expenditure: ${predicted_charge[0]:,.2f}")
```

## Future Improvements

- Add interaction terms (e.g. `age × smoker`, `bmi × smoker`) to capture compounding effects
- Explore regularized linear models (Ridge/Lasso) to guard against overfitting
- Add cross-validation for more robust performance estimates
- Build a simple web interface for premium estimation
- Log feature contributions per prediction for audit/regulatory purposes

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Rishabh2170**
