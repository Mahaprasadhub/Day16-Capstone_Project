# Capstone Mini-Project: Predict Customer Churn

## 1. Project Overview

Customer churn means a customer stops using a company's product or service.

The goal of this project is to build a **Machine Learning classification model** that predicts whether a customer is likely to:

- `0` = Stay
- `1` = Churn

The final selected model is a **tuned Gradient Boosting Classifier**.

## 2. Problem Statement

Losing customers directly affects a subscription-based company's revenue. Instead of waiting until customers leave, Machine Learning can identify customers who are at risk of churning.

The model looks for patterns such as:

- Low customer tenure
- High monthly charges
- Many support tickets
- High network latency
- A long time since the last login
- Month-to-month contracts

The model then returns a churn probability for each customer.

```text
Customer A -> Churn probability = 0.12
Customer B -> Churn probability = 0.84
```

Customer B is considered more likely to churn.

## 3. Complete Project Workflow

```text
Create Synthetic Dataset
        |
Explore the Data
        |
Detect and Remove Leakage
        |
Train / Validation / Test Split
        |
Preprocess the Features
        |
Compare Four ML Models
        |
Tune Gradient Boosting
        |
Evaluate on Test Data
        |
Select a Probability Threshold
        |
Interpret Feature Importance
```

## 4. Dataset

A synthetic customer dataset is created using:

```python
from sklearn.datasets import make_classification
```

The dataset contains **4,000 customer records**.

```python
N = 4000
```

The classes are intentionally imbalanced:

```python
weights=[0.76, 0.24]
```

This means most customers belong to the stay class, while churn customers form the minority class. The final proportions can vary slightly because `flip_y=0.13` adds label noise.

## 5. Features

### Numerical Features

| Feature | Meaning |
|---|---|
| `tenure_months` | Number of months the customer has used the service |
| `monthly_charges` | Amount paid by the customer every month |
| `support_tickets` | Number of support tickets raised |
| `avg_latency_ms` | Average service latency in milliseconds |
| `addons_count` | Number of additional services used |
| `days_since_login` | Days since the customer's last login |
| `total_charges` | Total amount paid by the customer |

`total_charges` is calculated as:

```python
total_charges = monthly_charges * tenure_months
```

### Categorical Features

| Feature | Possible Values |
|---|---|
| `contract_type` | Month-to-month, One year, Two year |
| `payment_method` | Electronic check, Mailed check, Bank transfer, Credit card |
| `internet_service` | Fiber optic, DSL, None |
| `has_dependents` | Yes, No |

### Target Variable

The target column is `churn`:

```text
0 = Customer stays
1 = Customer churns
```

Because there are two possible outcomes, this is a **binary classification problem**.

## 6. Exploratory Data Analysis

The first step is to check the class distribution:

```python
df["churn"].value_counts(normalize=True)
```

This shows the percentage of customers who stay and churn.

The average numerical values of both groups are compared using:

```python
df.groupby("churn")[num_cols].mean()
```

This helps answer questions such as:

- Do churned customers have lower tenure?
- Do they create more support tickets?
- Are their monthly charges higher?
- Do they have higher latency?
- Do they log in less often?

### Visualizations

The project creates two visualizations:

1. **Tenure by outcome:** compares tenure distributions for stay and churn customers.
2. **Churn rate by contract:** compares churn rates across contract types.

## 7. Data Leakage

Data leakage happens when the model is given information that would not realistically be available when making a prediction.

The dataset contains an `exit_survey_sent` column. This feature is generated almost directly from the churn outcome, so it provides the model with part of the answer.

The project checks numeric correlations with churn:

```python
leak_scores = (
    df.corr(numeric_only=True)["churn"]
      .drop("churn")
      .abs()
      .sort_values(ascending=False)
)
```

Features with an absolute correlation greater than `0.9` are treated as leakage suspects and removed:

```python
suspects = leak_scores[leak_scores > 0.9].index.tolist()
df_clean = df.drop(columns=suspects)
```

> A correlation above 0.9 does not always prove leakage. In a real project, leakage should also be identified using business knowledge and by checking when each feature becomes available.

## 8. Train, Validation and Test Split

After removing leakage, the data is divided into:

- **70% training data:** used to train the models
- **15% validation data:** used to evaluate the tuned model before final testing
- **15% test data:** used only for final evaluation

For 4,000 rows, this gives approximately:

```text
Training   -> 2,800 rows
Validation ->   600 rows
Test       ->   600 rows
```

The split uses:

```python
stratify=y
```

Stratification keeps approximately the same churn percentage in the training, validation and test sets.

## 9. Preprocessing Pipeline

The project automatically identifies numerical and categorical columns:

```python
num_features = X_train.select_dtypes("number").columns.tolist()
cat_features = X_train.select_dtypes("object").columns.tolist()
```

### StandardScaler

Numerical features are processed using:

```python
StandardScaler()
```

Standardization puts numerical features on a similar scale, with values roughly centered around a mean of 0 and a standard deviation of 1. This is especially useful for Logistic Regression and SVM.

### OneHotEncoder

Categorical features are processed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

For example, contract type is converted into machine-readable columns:

```text
Month-to-month -> 1 0 0
One year       -> 0 1 0
Two year       -> 0 0 1
```

`handle_unknown="ignore"` prevents prediction from failing if a new category appears later.

### ColumnTransformer

`ColumnTransformer` applies different preprocessing to different column types:

```text
Numerical columns   -> StandardScaler
Categorical columns -> OneHotEncoder
```

### Pipeline

A Scikit-learn `Pipeline` combines preprocessing and model training:

```text
Raw Customer Data
        |
Preprocessing
        |
Machine Learning Model
        |
Churn Probability
```

This ensures that the same transformations are used during training, cross-validation and prediction.

## 10. Models Compared

Four model families are compared.

### Logistic Regression

A simple and interpretable binary classification model. It is used as a strong baseline.

### Random Forest

Combines many decision trees and can learn nonlinear relationships.

### Gradient Boosting

Builds decision trees sequentially. Each new tree attempts to correct mistakes made by previous trees.

### Support Vector Machine

Uses an RBF kernel to learn a nonlinear boundary between stay and churn customers.

For Logistic Regression, Random Forest and SVM, the project uses:

```python
class_weight="balanced"
```

This gives more importance to the minority churn class.

## 11. Cross-Validation

The models are compared using:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Five-fold cross-validation evaluates each model five times using different validation folds. Stratification keeps the churn ratio similar in each fold.

## 12. Model Selection Metric: ROC-AUC

The models are compared using:

```python
scoring="roc_auc"
```

ROC-AUC measures how well the model ranks churn customers above stay customers across different thresholds.

A simple interpretation is:

```text
0.50 -> Random ranking
0.60 -> Some predictive power
0.70 -> Reasonable discrimination
0.80 -> Strong discrimination
0.90 -> Very strong discrimination
1.00 -> Perfect ranking
```

Higher is better. These values are only rough guidelines.

## 13. Hyperparameter Tuning

Gradient Boosting is tuned using `GridSearchCV`.

The search checks:

```python
param_grid = {
    "clf__n_estimators": [150, 300],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [2, 3],
}
```

### Parameter Meaning

- `n_estimators`: number of boosting trees
- `learning_rate`: contribution made by each new tree
- `max_depth`: maximum complexity of each tree

There are eight parameter combinations. Each is evaluated with five-fold cross-validation, giving 40 cross-validation fits before the winning configuration is refitted.

## 14. Validation Evaluation

The best model is evaluated on the validation set:

```python
val_auc = roc_auc_score(
    y_val,
    best_model.predict_proba(X_val)[:, 1]
)
```

This checks whether the tuned model generalizes before the untouched test set is used.

## 15. Final Test Evaluation

The model generates churn probabilities:

```python
proba = best_model.predict_proba(X_test)[:, 1]
```

The final evaluation includes:

- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix

## 16. Classification Threshold

The default threshold is `0.50`:

```text
Probability < 0.50  -> Stay
Probability >= 0.50 -> Churn
```

The project compares the following thresholds:

```text
0.50, 0.40, 0.35, 0.30, 0.25
```

### Precision

Precision answers:

> Of all customers predicted to churn, how many actually churned?

```text
Precision = TP / (TP + FP)
```

### Recall

Recall answers:

> Of all customers who actually churned, how many did the model identify?

```text
Recall = TP / (TP + FN)
```

### F1-Score

F1-score balances precision and recall:

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

## 17. Why Use a Threshold of 0.30?

The project selects an operating threshold of `0.30` to favor recall.

For example, if a customer has a churn probability of `0.38`:

```text
At threshold 0.50 -> Predicted Stay
At threshold 0.30 -> Predicted Churn
```

Lowering the threshold usually:

- Increases recall
- Decreases precision
- Produces more churn alerts

This can be useful when missing an actual churner is more costly than contacting a customer who would have stayed.

The best threshold should ultimately be selected using actual business costs and retention capacity.

## 18. Confusion Matrix

```text
                 Predicted
               Stay   Churn
Actual Stay     TN      FP
Actual Churn    FN      TP
```

- **True Positive:** customer churned and was predicted to churn
- **True Negative:** customer stayed and was predicted to stay
- **False Positive:** customer stayed but was predicted to churn
- **False Negative:** customer churned but was predicted to stay

False negatives are important because they represent churners the business failed to identify.

## 19. Model Interpretation

### Permutation Importance

The tuned Gradient Boosting model is interpreted using:

```python
permutation_importance()
```

The method works by randomly shuffling one feature at a time and measuring how much model performance decreases. A large decrease means the model relies strongly on that feature.

The exact top drivers are taken from the program output:

```python
imp.head(3)
```

### Logistic Regression Coefficients

A Logistic Regression model is also trained for interpretation.

```text
Positive coefficient -> Associated with higher predicted churn odds
Negative coefficient -> Associated with lower predicted churn odds
```

These values show model associations. They do not prove that a feature causes churn.

## 20. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations and data generation |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Scikit-learn | Preprocessing, models, tuning and evaluation |
| Pipeline | Combines preprocessing and model training |
| ColumnTransformer | Applies different preprocessing by column type |
| GridSearchCV | Hyperparameter tuning |
| Permutation importance | Model interpretation |

## 21. Key Concepts Demonstrated

- Binary classification
- Customer churn prediction
- Exploratory Data Analysis
- Numerical and categorical features
- Class imbalance
- Data leakage
- Stratified data splitting
- Standardization
- One-hot encoding
- Machine Learning pipelines
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- Cross-validation
- ROC-AUC
- Hyperparameter tuning
- Confusion matrix
- Precision, recall and F1-score
- Probability threshold selection
- Feature importance
- Model interpretation

## 22. Business Value

The model can identify customers who are at a higher risk of leaving. The company could prioritize these customers for actions such as:

- Proactive customer support
- Service-quality investigation
- Contract upgrade offers
- Targeted discounts
- Personalized retention campaigns

The business goal is not only to predict churn. It is to **identify at-risk customers early enough for the business to take action**.

Because this project uses synthetic data, it demonstrates the Machine Learning workflow rather than proving real-world business impact.

## 23. Important Design Decisions

### Why remove `exit_survey_sent`?

It is almost determined by the churn outcome and creates target leakage.

### Why use stratified splitting?

Churn is the minority class, so similar class proportions are needed in every data split.

### Why use a Pipeline?

It makes preprocessing consistent and prevents accidental leakage during cross-validation.

### Why compare multiple models?

The best algorithm should be selected using evaluation results rather than assumed beforehand.

### Why use ROC-AUC?

It compares probability ranking across thresholds and is useful for imbalanced classification.

### Why focus on recall?

The business may care more about finding as many real churners as possible.

### Why lower the threshold?

A lower threshold normally improves churn recall, but it also creates more false-positive alerts.

## 24. How to Run the Project

### Install the Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run the Notebook or Python File

If the code is in a Python file:

```bash
python customer_churn.py
```

If it is in a Jupyter Notebook, run the cells from top to bottom.

## 25. Short Explanation for an Interview or SME Discussion

> My capstone project is a customer churn prediction problem. The objective is to identify customers who are likely to leave a service. I worked with a synthetic dataset containing 4,000 customers with numerical features such as tenure, monthly charges, support tickets, latency and days since login, along with categorical features such as contract type, payment method and internet service.
>
> I first performed exploratory data analysis and checked for target leakage. An exit-survey feature was strongly related to churn because it was generated using the outcome itself, so I removed leakage features before training.
>
> I then divided the data into training, validation and test sets using stratified splitting. I created a preprocessing pipeline in which numerical features are standardized using StandardScaler and categorical features are converted using OneHotEncoder.
>
> I compared Logistic Regression, Random Forest, Gradient Boosting and SVM using five-fold stratified cross-validation and ROC-AUC. I then tuned Gradient Boosting using GridSearchCV.
>
> Finally, I evaluated the selected model using ROC-AUC, precision, recall, F1-score and a confusion matrix. I tested different probability thresholds and selected 0.30 to favor churn recall while accepting lower precision. I also used permutation importance and Logistic Regression coefficients to interpret the model.

## 26. One-Minute Revision

```text
Problem     -> Predict customer churn

Target      -> 0 = Stay, 1 = Churn

Data        -> 4,000 synthetic customers

Split       -> 70% training, 15% validation, 15% test

Leakage     -> exit_survey_sent is detected and removed

Numeric     -> StandardScaler

Categorical -> OneHotEncoder

Models      -> Logistic Regression, Random Forest,
               Gradient Boosting and SVM

Validation  -> Five-fold Stratified Cross-Validation

Selection   -> ROC-AUC

Tuning      -> GridSearchCV

Final Model -> Tuned Gradient Boosting

Metrics     -> ROC-AUC, precision, recall,
               F1-score and confusion matrix

Threshold   -> 0.30 to favor churn recall

Explanation -> Permutation importance and
               Logistic Regression coefficients

Business    -> Find high-risk customers early so
               retention actions can be taken
```

## 27. Final Takeaway

This project is not only about training a Gradient Boosting model. It demonstrates an end-to-end Machine Learning process:

```text
Understand the data
-> Detect leakage
-> Split the data correctly
-> Preprocess the features
-> Compare models
-> Tune the selected model
-> Evaluate on unseen data
-> Select a business threshold
-> Explain the predictions
```

This complete workflow is the strongest part of the project to explain during a technical discussion.
