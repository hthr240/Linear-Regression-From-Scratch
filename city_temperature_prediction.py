
import pandas as pd 
import matplotlib.pyplot as plt
from linear_regression import *
from polynomial_fitting import *


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()

    # remove invalid (not possible rows)
    df = df[(df['Temp'] > -30) & (df['Temp'] < 50)]

    # Add DayOfYear column
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    return df

def plot_day_temp_by_day_of_year(df,country):

    plt.figure(figsize=(12, 6))

    # scatter for each year 
    for year in sorted(df["Year"].unique()):
        subset = df[df["Year"] == year]
        plt.scatter(subset["DayOfYear"], 
                    subset["Temp"], 
                    label=year, 
                    alpha=0.5, 
                    s=10)
    
    plt.title(f"Daily Temperature in {country} by Day of Year")
    plt.xlabel("Day of Year")
    plt.ylabel("Temperature (°C)")
    plt.legend(title="Year", bbox_to_anchor=(1, 0), loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{country}_day_year_temp.png')

def plot_day_temp_std_month(df, country):

    # group by months and get std for each group
    months_std = df.groupby("Month")["Temp"].agg(std="std").reset_index()

    # plot the stds
    plt.figure(figsize=(12, 6))
    plt.bar(months_std["Month"], 
            months_std["std"], 
            color='darkcyan')

    plt.title("Monthly std of Daily Temperatures in Israel")
    plt.xlabel("Month")
    plt.ylabel("Temperature Std Dev (°C)")
    plt.xticks(months_std["Month"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{country}_day_temp_month_std.png')

def plot_mean_std_monthly_temp_by_country(df):

    # group country and motnh
    df_groups = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()
    plt.figure(figsize=(12, 6))

    # plot error bar for each country
    for country in df_groups['Country'].unique():

        country_data = df_groups[df_groups['Country'] == country]
        plt.errorbar(country_data['Month'],
                     country_data['mean'],
                     yerr=country_data['std'],
                     label=country,
                     capsize=8,
                     marker='o',
                     linestyle='-',
                     linewidth=3,
                     elinewidth=7,
                     alpha=0.8)
        
        plt.xlabel('Month')
        plt.ylabel('Average Temperature (°C)')
        plt.title('Average Monthly Temperature with Error Bars by Country')
        plt.legend(title='Country')
        plt.grid(True)

    plt.savefig('mean_std_month_temp.png')

def train_test_split(X:np.ndarray, y, train_size=0.75, seed=26):

    np.random.seed(seed)
    indices = np.random.permutation(X.index)
    train_end = int(train_size * len(indices))

    train_idx = indices[:train_end]
    test_idx = indices[train_end:]

    return X.loc[train_idx], y.loc[train_idx], X.loc[test_idx], y.loc[test_idx]

def compute_loss_for_ks(train_X, train_y, test_X, test_y):
    
    # array for losses
    losses = np.zeros(10, dtype=np.float32)
    # array for k values
    k_values = range(1, 11)

    # plot for each k
    for k in k_values:

        poly_fit = PolynomialFitting(k)
        poly_fit.fit(train_X['DayOfYear'].to_numpy(), train_y.to_numpy())
        loss = poly_fit.loss(test_X['DayOfYear'].to_numpy(), test_y.to_numpy())
        losses[k-1] = np.round(loss,2)
    
    for k, loss in zip(k_values, losses): 
        print(f"k = {k}, Test Error (Loss) = {loss:.2f}")

    plt.figure(figsize=(12, 6))
    bars = plt.bar(k_values, losses, color='darkcyan')

    # Add loss value on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, 
                 height, 
                 f'{height:.2f}',
                 ha='center', 
                 va='bottom', 
                 fontsize=10)

    plt.xlabel("Polynomial Degree (k)")
    plt.ylabel("Test Loss (MSE)")
    plt.title("Test Loss for Polynomial Degrees (k=1 to 10)")
    plt.xticks(k_values)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('temp_loss_for_ks.png')

def plot_countries_temp_err_from_best_k_fitted_israel_model(data, israel_data):
    
    poly_fit = PolynomialFitting(5)
    poly_fit.fit(israel_data['DayOfYear'].to_numpy(), israel_data['Temp'].to_numpy())

    countries = data['Country'].unique()
    country_errors = {}

    for country in countries:
        if country == "Israel":
            continue
        country_data = data[data['Country'] == country]
        X_country = country_data['DayOfYear'].to_numpy()
        y_country = country_data['Temp'].to_numpy()
        
        # Predict using the model fitted on Israel's data
        predictions = poly_fit.predict(X_country)
        
        # Calculate MSE for the country
        mse = np.mean((y_country - predictions) ** 2)
        country_errors[country] = mse

    countries_sorted = sorted(country_errors.keys(), key=lambda x: country_errors[x])
    mse_values = [country_errors[country] for country in countries_sorted]
    
    # Plot the MSE for each country
    plt.figure(figsize=(12, 6))
    plt.bar(countries_sorted, mse_values, color='darkcyan')

    # Add MSE value on top of each bar
    for i, mse in enumerate(mse_values):
        plt.text(i, mse + 10, f'{mse:.2f}', ha='center', fontsize=10)

    plt.ylim(0,150)
    plt.xlabel("Country")
    plt.ylabel("Test Error (MSE)")
    plt.title(f"Test Error for Countries Based on Israel's Model (k={5})")
    plt.xticks(rotation=90)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('countries_mse_based_on_israel_model.png')


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    il_df = df[df['Country'] == 'Israel']
    plot_day_temp_by_day_of_year(il_df,'israel')
    plot_day_temp_std_month(il_df,'israel')

    # Question 4 - Exploring differences between countries
    plot_mean_std_monthly_temp_by_country(df)

    # Question 5 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = train_test_split(il_df.drop('Temp', axis=1), il_df.Temp)
    compute_loss_for_ks(train_X, train_y, test_X, test_y)

    # Question 6 - Evaluating fitted model on different countries
    plot_countries_temp_err_from_best_k_fitted_israel_model(df, il_df)
