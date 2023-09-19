import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import joblib



pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./beers/recipeData.csv", encoding="latin1")
orig_leng = len(df)
df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale", "FG", "Color", "BrewMethod"])
df.dropna(inplace=True)
ser = df.isna().mean() * 100


df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
df = df[(df["IBU"] <= 150) & (df["IBU"] > 0)]  # IBU max == 150 selon wikipedia
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]
df['OG2'] = df['OG'] ** 2

new_len = len(df)
one_hot = pd.get_dummies(df[["StyleID"]], columns=["StyleID"])
df = df.join(one_hot)
df = df.drop(
    columns=["StyleID"]
)

df_X_ABV = df.drop(columns=["ABV", "IBU"])
df_X_IBU = df.drop(columns=['ABV', 'IBU'])
df_y_ABV = df["ABV"]
df_y_IBU = df["IBU"]
"""
X_train_ABV, X_test_ABV, y_train_ABV, y_test_ABV = train_test_split(df_X_ABV, df_y_ABV)
X_train_IBU, X_test_IBU, y_train_IBU, y_test_IBU = train_test_split(df_X_IBU, df_y_IBU)

reg_ABV = LinearRegression().fit(X_train_ABV, y_train_ABV)
reg_IBU = LinearRegression().fit(X_train_IBU, y_train_IBU)
regm_ABV = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=500).fit(X_train_ABV, y_train_ABV)
regm_IBU = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=500).fit(X_train_IBU, y_train_IBU)
regr_ABV = RandomForestRegressor().fit(X_train_ABV, y_train_ABV)
regr_IBU = RandomForestRegressor().fit(X_train_IBU, y_train_IBU)

# print("Erreur MSE RL_ABV :", mean_squared_error(y_test_ABV, reg_ABV.predict(X_test_ABV)))
# print("Erreur MSE RL_IBU :", mean_squared_error(y_test_IBU, reg_IBU.predict(X_test_IBU)))
print("Erreur MSE RL_ABV :", mean_squared_error(y_test_ABV, regm_ABV.predict(X_test_ABV)))
print("Erreur MSE RL_IBU :", mean_squared_error(y_test_IBU, regm_IBU.predict(X_test_IBU)))
print("Erreur MSE RM_ABV :", mean_squared_error(y_test_ABV, regr_ABV.predict(X_test_ABV)))
print("Erreur MSE RM_IBU :", mean_squared_error(y_test_IBU, regr_IBU.predict(X_test_IBU)))
print("Erreur MSE RA_ABV :", mean_squared_error(y_test_ABV, reg_ABV.predict(X_test_ABV)))
print("Erreur MSE RA_IBU :", mean_squared_error(y_test_IBU, reg_IBU.predict(X_test_IBU)))


"""

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Split the data into training and testing sets
X_train_IBU, X_test_IBU, y_train_IBU, y_test_IBU = train_test_split(df_X_IBU, df_y_IBU)

# Create the models
rf = RandomForestRegressor()

# Create HalvingGridSearchCV instances for each model
search_rf = HalvingGridSearchCV(rf, param_grid_rf, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the models with hyperparameter tuning
search_rf.fit(X_train_IBU, y_train_IBU)

# Get the best models with optimized hyperparameters
best_rf = search_rf.best_estimator_

# Evaluate the models
mse_rf = mean_squared_error(y_test_IBU, best_rf.predict(X_test_IBU))

print("Best Random Forest MSE:", mse_rf)

