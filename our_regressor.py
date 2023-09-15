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
df['OGPoly'] = df["OG"] + df['OG'] ** 2

new_len = len(df)
# print(new_len / orig_leng)
one_hot = pd.get_dummies(df[["StyleID"]], columns=["StyleID"])

df_X = df.drop(columns=["ABV", "IBU"])
df_X_IBU = df.drop(columns=['Size(L)', 'ABV', 'IBU', 'BoilSize', 'BoilTime'])
df_y_ABV = df["ABV"]
df_y_IBU = df["IBU"]

X_train_ABV, X_test_ABV, y_train_ABV, y_test_ABV = train_test_split(df_X, df_y_ABV)
X_train_IBU, X_test_IBU, y_train_IBU, y_test_IBU = train_test_split(df_X_IBU, df_y_IBU)

regr_ABV = RandomForestRegressor().fit(X_train_ABV, y_train_ABV)
regr_IBU = RandomForestRegressor(min_samples_leaf=4, max_depth=14, n_estimators=300, min_samples_split=24).fit(X_train_IBU, y_train_IBU)

# Save your trained models
joblib.dump(regr_ABV, 'models/random_forest_ABV.pkl')
joblib.dump(regr_IBU, 'models/random_forest_IBU.pkl')
