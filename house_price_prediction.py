
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import NoReturn
from linear_regression import LinearRegression


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    # change floor type to float
    X["floors"] = X["floors"].astype(float)
    
    # invalid rows
    invalid_rows = ((X["yr_built"] > X["yr_renovated"]) | (X["sqft_living"] > X["sqft_lot"]) | 
                    (X["bathrooms"] <= 0) | (X["bedrooms"] < 0) | (X["sqft_living"] <= 0) |
                    (X["sqft_lot"] <= 0) | (X["floors"] <= 0) | (X["sqft_above"] <= 0) | X["yr_built"] <= 0 | 
                    (X["sqft_living15"] <= 0) | (X["sqft_lot15"] <= 0) | (X["sqft_basement"] < 0) |
                    (X["yr_renovated"] < 0) | (~X["waterfront"].isin((0,1))) | (~X["view"].isin(range(5))) |
                    (~X["condition"].isin(range(1,6))) | (~X["grade"].isin(range(1,14))) |
                    (X.isnull().any(axis=1)) | (y <= 0) | (y.isnull()))

    # get valid rows
    valid_rows = ~invalid_rows
    X = X[valid_rows].reset_index(drop=True)
    y = y[valid_rows].reset_index(drop=True)

    # TODO add beneficial columns
    X["last_update"] = X[["yr_built", "yr_renovated"]].max(axis=1)
    # remove unnecessery columns
    X = X.drop(columns=["date","id","long","lat","yr_built","yr_renovated"])

    return X, y
    
def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """

    X["last_update"] = X[["yr_built", "yr_renovated"]].max(axis=1)
    X = X.drop(columns=["date","id","long","lat","yr_built","yr_renovated"])

    return X

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "eval_graphs") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # create the file if not exist
    os.makedirs(output_path,exist_ok=True)

    for col in X.columns:

        # compute the pearson correlation
        cov = np.cov(X[col],y)[0,1]
        std_x = np.std(X[col])
        std_y = np.std(y)
        pearson_correlation = cov / (std_x * std_y)

        # create plot
        plt.figure(figsize=(12,6))
        plt.scatter(X[col], y, alpha=0.5, color="darkcyan")

        m, b = np.polyfit(X[col], y, 1)
        plt.plot(X[col], m * X[col] + b, color='lightcoral')

        # plot title and labels
        plt.title(f"{col} - price\nPearson Correlation = {pearson_correlation}")
        plt.xlabel(f"{col} values")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()

        # save plot
        file_name = f"{output_path}/{col}_pearson_correlation.png"
        plt.savefig(file_name)
        plt.close()
        
def train_test_split(X, y, train_size=0.75, seed=26):

    # set seed and randomize data
    np.random.seed(seed)
    indices = np.random.permutation(X.index)
    train_end = int(train_size * len(indices))

    train_idx = indices[:train_end]
    test_idx = indices[train_end:]

    return X.loc[train_idx], y.loc[train_idx], X.loc[test_idx], y.loc[test_idx]

def measure_loss(train_X, train_y, test_X, test_y):
    # init noArray with zeroes for result
    results = np.zeros((91, 10))

    for i, p in enumerate(range(10,101)):

        frac = p / 100.0
        for j in range(10):

            # make sample portions
            sample_X = train_X.sample(frac=frac)
            sample_y = train_y[sample_X.index]

            # fit and calculate loss
            linear_regression = LinearRegression()
            linear_regression.fit(sample_X.to_numpy(), sample_y.to_numpy())
            loss = linear_regression.loss(test_X.to_numpy(), test_y.to_numpy())

            # save loss result in relevant cell
            results[i,j] = loss

    return results

def plot_loss(mean_loss, std_loss):
    percentage_range = range(10, 101)

    # plot mean test loss with error
    plt.figure(figsize=(12, 6))
    plt.plot(percentage_range, mean_loss, marker='o', color='red', label='Mean Test Loss')
    plt.fill_between(percentage_range, mean_loss - 2 * std_loss, mean_loss + 2 * std_loss,
                     color='lightcoral', alpha=0.4, label='±2 Std Dev')

    # Add labels and title
    plt.title('Test Loss as a Function of Training Set Size')
    plt.xlabel('Percentage of Training Set Used')
    plt.ylabel('Mean Squared Error (MSE) on Test Set')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save plot
    plt.savefig('loss_plot.png')
    plt.close()

if __name__ == '__main__':

    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test

    train_X, train_y, test_X, test_y = train_test_split(X,y)

    # Question 3 - preprocessing of housing prices train dataset

    train_X, train_y = preprocess_train(train_X, train_y)

    # Question 4 - Feature evaluation of train dataset with respect to response

    feature_evaluation(train_X, train_y)

    # Question 5 - preprocess the test data
    
    test_X = preprocess_test(test_X)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    # reorder and adjust missing values
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)
    # get loss
    loss_results = measure_loss(train_X, train_y, test_X, test_y)
    # get mean loss and std loss
    mean_loss, std_loss = loss_results.mean(axis=1), loss_results.std(axis=1)
    # plot loss
    plot_loss(mean_loss, std_loss)
    