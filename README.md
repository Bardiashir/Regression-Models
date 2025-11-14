# Regression-Models
# Regression Models (Linear &amp; Logistic)  This project demonstrates two machine learning models using Python, pandas, scikit-learn, and matplotlib.   Both scripts are based on realistic scenarios: one predicting restaurant profitability, and the other predicting hiring decisions.


---

## 📈 Linear Regression — Predicting Restaurant Profit

**File:** `LinearRegression.py`

This model estimates how the profit of a restaurant relates to the population of the city where it operates.

### 🔍 Real-World Scenario  
A restaurant chain wants to know whether opening a restaurant in a new city will be profitable.  
They have historical data from multiple cities:

- **X:** population size  
- **y:** restaurant profit/loss  

The script:

- Loads and visualizes the population–profit data  
- Uses linear regression to model the relationship  
- Prints the learned parameters (intercept & coefficient)  
- Predicts the profit for a city with 18 habitants  
- Plots the regression line over the original data  

This provides a simple predictive tool for evaluating new restaurant locations.

---

## 🧠 Logistic Regression — Predicting Hiring Decisions

**File:** `LogisticRegression.py`

This model predicts whether an applicant is likely to be **hired (1)** or **rejected (0)** after a technical interview.

### 🔍 Real-World Scenario  
A recruiter collected data over several years, including:

- **Score1:** result of technical question 1  
- **Score2:** result of technical question 2  
- **y:** hiring outcome (0 or 1)  

The script:

- Loads and plots the data with different markers for each class  
- Trains a logistic regression classifier  
- Predicts labels on the training set  
- Visualizes how well the classifier separates the two classes  
- Highlights errors through mismatched markers/colors  

This shows how logistic regression can model decision boundaries for binary outcomes.

---

## 📦 Requirements

Install dependencies with:

```
pip install -r requirements.txt
