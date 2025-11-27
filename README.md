A custom implementation of **Ordinary Least Squares (OLS) Linear Regression** built from scratch using NumPy. This project analyzes housing prices and temperature trends, demonstrating the mathematical foundations of machine learning without relying on high-level frameworks like Scikit-Learn.

## 🚀 Key Features
* **Algorithm Implementation:** Implemented the Closed-Form Solution (Normal Equation) for Linear Regression: $\hat{w} = (X^T X)^{-1} X^T y$.
* **Data Preprocessing:** Advanced data cleaning using **Pandas**, including outlier detection, feature engineering (e.g., `last_update` year), and invalid row filtering.
* **Feature Selection:** Automated Pearson Correlation analysis to identify significant predictors.
* **Bias-Variance Analysis:** Visualized the impact of training set size on model loss (MSE) and confidence intervals.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** NumPy (Linear Algebra), Pandas (Data Manipulation), Matplotlib (Visualization).
* **Concepts:** Supervised Learning, OLS, Polynomial Regression, Overfitting/Underfitting.

## 📂 Project Structure
* `linear_regression.py`: The custom estimator class implementing `fit()`, `predict()`, and `loss()`.
* `polynomial_fitting.py`: Extension of the linear model to handle non-linear relationships via polynomial feature transformation.
* `house_price_prediction.py`: Script for training the model on the Housing Prices dataset.
* `city_temperature_prediction.py`: Analysis script for polynomial fitting on weather data.

## 🧠 Algorithmic Implementation
Unlike standard library calls, this project manually handles the linear algebra:
1. **Design Matrix:** Appends a bias term (column of ones) to the input features.
2. **Optimization:** Calculates weights using the Singular Value Decomposition (SVD) pseudo-inverse method (`np.linalg.pinv`) for numerical stability.
3. **Evaluation:** Computes Mean Squared Error (MSE) to quantify model performance.

## 📊 Results
* **Feature Analysis:** Identified high correlation between `sqft_living` and price, while `condition` showed lower linear correlation.
* **Polynomial Fitting:** Modeled daily average temperatures, identifying optimal polynomial degrees to balance bias and variance.
