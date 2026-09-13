\# Customer Retention Intelligence



An end-to-end machine learning project for predicting customer churn, explaining churn risk, and converting model predictions into actionable customer retention strategies.



The project moves beyond binary classification by combining:



\*\*Data Analysis → Churn Prediction → Model Explainability → Risk Segmentation → Retention Actions\*\*



\---



\## Project Overview



Customer churn is not only a classification problem. A useful churn system should also answer:



\- Which customers are likely to leave?

\- What characteristics are associated with churn?

\- Why does the model consider a specific customer high risk?

\- How should the business prioritize retention efforts?

\- What decision threshold makes sense for the business objective?



This project addresses each of these questions using the IBM Telco Customer Churn dataset.



\---



\## Machine Learning Workflow



\### 00 — Data Acquisition \& EDA



\- Download and inspect the raw dataset

\- Identify data quality issues

\- Clean `TotalCharges`

\- Analyze churn distribution

\- Explore churn patterns across:

&#x20; - Contract type

&#x20; - Customer tenure

&#x20; - Payment method

&#x20; - Internet service

&#x20; - Tech support

&#x20; - Online security

&#x20; - Monthly charges



Key EDA findings showed particularly high churn among newer customers, month-to-month subscribers, fiber optic users, electronic check users, and customers without support services.



\### 01 — Logistic Regression Baseline



A preprocessing and Logistic Regression pipeline was built using:



\- `StandardScaler` for continuous numerical features

\- `OneHotEncoder` for categorical features

\- passthrough for binary features

\- stratified train-test splitting



Baseline test performance:



| Metric | Score |

|---|---:|

| Accuracy | 0.803 |

| Precision | 0.652 |

| Recall | 0.556 |

| F1 | 0.600 |

| ROC-AUC | 0.842 |



Threshold analysis demonstrated that the default probability threshold of `0.50` is not necessarily optimal for customer retention.



\---



\## Tree-Based Models



The Logistic Regression baseline was compared against:



\- Decision Tree

\- Random Forest

\- Gradient Boosting



Five-fold stratified cross-validation produced:



| Model | Mean ROC-AUC | Std ROC-AUC |

|---|---:|---:|

| Logistic Regression | 0.845 | 0.013 |

| Gradient Boosting | 0.847 | 0.011 |



Gradient Boosting was then tuned using cross-validation.



Best parameters:



```python

learning\_rate = 0.05

max\_depth = 2

min\_samples\_leaf = 20

n\_estimators = 200

```



The tuned model achieved approximately:



\*\*Test ROC-AUC: 0.846\*\*



\---



\## Decision Threshold Optimization



For retention applications, missing a future churner may be more expensive than contacting a customer who would not have churned.



The tuned Gradient Boosting model produced:



| Threshold | Precision | Recall | F1 |

|---:|---:|---:|---:|

| 0.50 | 0.661 | 0.505 | 0.573 |

| 0.40 | 0.592 | 0.660 | 0.625 |

| \*\*0.35\*\* | \*\*0.569\*\* | \*\*0.717\*\* | \*\*0.634\*\* |

| 0.30 | 0.530 | 0.767 | 0.627 |



A threshold of \*\*0.35\*\* was selected as a balanced retention strategy.



\---



\## Model Explainability



Two forms of model explainability were used.



\### Global Explainability



Gradient Boosting identified several important predictors of churn, including:



\- Month-to-month contracts

\- Customer tenure

\- Fiber optic internet

\- Lack of online security

\- Lack of technical support

\- Electronic check payment

\- Monthly and total charges



Logistic Regression coefficients were also analyzed to understand the direction of relationships between customer characteristics and churn.



\### Customer-Level Explainability



SHAP was used to explain individual churn predictions.



For one high-risk customer, the model identified factors such as:



\- tenure of only one month

\- month-to-month contract

\- fiber optic service

\- high monthly charges

\- no technical support

\- no online security

\- electronic check payment



as major contributors to a churn probability above 90%.



\---



\## Retention Intelligence



The final model was applied to customers using a business decision threshold of `0.35`.



Out of \*\*7,043 customers\*\*:



\- \*\*2,270 customers\*\*

\- approximately \*\*32.2%\*\*



were identified as potential retention targets.



Customers were segmented into:



```text

Low Risk

Moderate Risk

High Risk

Very High Risk

```



Predictions were then converted into recommended actions such as:



```text

Very High Risk

→ Priority outreach + long-term contract incentive



High Risk + No Tech Support

→ Offer technical support / service assistance



High Risk

→ Targeted retention offer



Moderate Risk

→ Monitor and send engagement offer



Low Risk

→ No immediate action

```



The final pipeline therefore produces more than a churn prediction:



```text

Customer Data

&#x20;     ↓

Churn Probability

&#x20;     ↓

Risk Segment

&#x20;     ↓

Retention Target

&#x20;     ↓

Recommended Action

```



\---



\## Project Structure



```text

customer-retention-intelligence/

│

├── notebooks/

│   ├── 00\_data\_acquisition\_eda.ipynb

│   ├── 01\_churn\_baseline.ipynb

│   ├── 02\_tree\_models.ipynb

│   ├── 03\_model\_explainability.ipynb

│   └── 04\_retention\_strategy.ipynb

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│   └── churn\_gradient\_boosting.joblib

│

├── outputs/

│   └── retention\_target\_list.csv

│

├── src/

├── app/

├── .gitignore

└── README.md

```



\---



\## Tech Stack



\- Python

\- Pandas

\- Matplotlib

\- scikit-learn

\- SHAP

\- Joblib

\- Jupyter Notebook



\---



\## Key Takeaway



The project demonstrates that successful churn modeling requires more than maximizing model accuracy.



A useful retention system must combine predictive performance with model explainability, probability calibration and threshold selection, customer segmentation, and business-oriented decision rules.



The result is an interpretable machine learning workflow that turns churn probabilities into prioritized retention actions.



\---



\## Next Step



The next development step is to expose the trained pipeline through a lightweight Streamlit application where customer information can be entered to generate:



\- churn probability

\- risk segment

\- retention recommendation

\- explanation of the prediction

