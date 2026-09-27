# Customer Churn Prediction

An end-to-end machine learning system that predicts customer churn using the Telco Customer Churn dataset, deployed as a REST API with a Streamlit frontend.

## Problem
Telecom companies lose revenue when customers churn. This project identifies customers at high risk of churning so retention efforts can be targeted effectively.

## Approach
- Cleaned and explored the Telco Customer Churn dataset (7,000+ rows)
- Found key churn drivers through EDA: contract type, tenure, and monthly charges
- Addressed class imbalance (~27% churn rate) using `class_weight='balanced'`
- Compared Logistic Regression and Random Forest models
- Selected [your final model] based on recall for the churn class, since missing a churner is costlier than a false alarm
- Deployed the model via FastAPI, with a Streamlit UI for live predictions

## Key Insights
- Month-to-month contracts churn at a significantly higher rate than one/two-year contracts
- New customers (low tenure) are far more likely to churn than long-term customers
- Top predictive features: TotalCharges, tenure, MonthlyCharges

## Results
- Final model: [Random Forest / Logistic Regression]
- Accuracy: [your number]%
- Recall (Churn class): [your number]%
- [Add your confusion matrix or a screenshot of the classification report]

## Tech Stack
Python, Pandas, Scikit-learn, FastAPI, Streamlit, Joblib

## How to Run
1. Clone this repo
2. Create a virtual environment and install requirements:
pip install -r requirements.txt

3. Run the API:

uvicorn app:app --reload

4. In a separate terminal, run the frontend:

streamlit run frontend.py