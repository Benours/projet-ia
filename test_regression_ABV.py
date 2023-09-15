import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./beers/recipeData.csv", encoding="latin1")
orig_leng = len(df)
df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale", "FG", "Color"])
df.dropna(inplace=True)
ser = df.isna().mean() * 100


df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
df = df[df["IBU"] <= 150]
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]

new_len = len(df)

one_hot = pd.get_dummies(df["BrewMethod"])
df = df.drop(columns=["BrewMethod"])
df = df.join(one_hot)


df_X = df.drop(columns=["ABV", "IBU"])
df_y = df[["ABV", "IBU"]]
df_y_ABV = df["ABV"]
df_y_IBU = df["IBU"]


X_train_ABV, X_test_ABV, y_train_ABV, y_test_ABV = train_test_split(df_X, df_y_ABV)
X_train_IBU, X_test_IBU, y_train_IBU, y_test_IBU = train_test_split(df_X, df_y_IBU)

regr_ABV = RandomForestRegressor().fit(X_train_ABV, y_train_ABV)
regr_IBU = RandomForestRegressor().fit(X_train_IBU, y_train_IBU)

print("Erreur MSE RA_ABV :", mean_squared_error(y_test_ABV, regr_ABV.predict(X_test_ABV)))
print("Erreur MSE RA_IBU :", mean_squared_error(y_test_IBU, regr_IBU.predict(X_test_IBU)))
