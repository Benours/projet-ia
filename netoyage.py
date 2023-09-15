import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./beers/recipeData.csv", encoding="latin1")
orig_leng = len(df)
"""
df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale"])  # PrimaryTemp
"""
df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale", "FG", "Color"])
df.dropna(inplace=True)
ser = df.isna().mean() * 100
# print(df)


# aze = df["Size(L)"].quantile(0.95)
df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
# df = df[df["FG"] <= df["FG"].quantile(0.95)]
df = df[(df["IBU"] <= 150) & (df["IBU"] > 0)]  # IBU max == 150 selon wikipedia
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]
# df["WortDelta"] = df["OG"] - df["FG"]
df['OG2'] = df['OG'] ** 2
df['BoilGravity2'] = df['BoilGravity'] ** 2


# hist = df.hist(bins=50, log=True)
# print(df.describe(include='all'))
"""
new_len = len(df)
# print(new_len / orig_leng)
one_hot = pd.get_dummies(df[["StyleID"]], columns=["StyleID"])
print(one_hot)
"""
df = df.drop(columns=["BrewMethod", "StyleID"])
# df = df.join(one_hot)

"""
# Calculer la matrice de corrélation (ici, la méthode Pearson)
corr_matrix = df.corr(method='pearson')

# Créer la figure et l'axe
fig, ax = plt.subplots(figsize=(8, 6))

# Afficher la matrice de corrélation avec plt.matshow()
cax = ax.matshow(corr_matrix, cmap='coolwarm')

# Ajouter une barre de couleur
fig.colorbar(cax)

# Définir les étiquettes des axes
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45)
ax.set_yticklabels(corr_matrix.columns, rotation=45)

fig, axes = plt.subplots(nrows=10, ncols=20)
for ax, col in zip(axes.flatten(), df.columns):
    ax.scatter(df[col], df["IBU"])
    ax.set_title("Column: %s" % col)
"""
df = df.drop(columns=["ABV"])
# Calculer la matrice de corrélation entre IBU et les autres variables
correlation_with_ibu = df.corr()["IBU"]

# Afficher les corrélations avec IBU
print(correlation_with_ibu)

"""
fig, axes = plt.subplots(12, 15)
plt.subplots_adjust(wspace=10, hspace=10)
for ax, style in zip(axes.flatten(), one_hot.columns):
    df.hist(column="IBU", bins=50, ax=ax, facecolor='#00FF00', density=True)
    df[df[style]].hist(column="IBU", bins=50, ax=ax, facecolor='#FF000075', density=True)
    ax.set_title(style)

# Afficher le plot
plt.show()
"""
"""
df_X = df.drop(columns=["ABV", "IBU"])
# df_y = df[["ABV", "IBU"]]
# df_y_ABV = df["ABV"]
df_y_IBU = df["IBU"]

print("Columns :", df.columns)


# X_train_ABV, X_test_ABV, y_train_ABV, y_test_ABV = train_test_split(df_X, df_y_ABV)
X_train_IBU, X_test_IBU, y_train_IBU, y_test_IBU = train_test_split(df_X, df_y_IBU)

# reg_ABV = LinearRegression().fit(X_train_ABV, y_train_ABV)
reg_IBU = LinearRegression().fit(X_train_IBU, y_train_IBU)
# regr_ABV = RandomForestRegressor().fit(X_train_ABV, y_train_ABV)
regr_IBU = RandomForestRegressor().fit(X_train_IBU, y_train_IBU)

print("Indicateur RL_IBU :", reg_IBU.coef_)
print("Indicateur RA_IBU :", regr_IBU.feature_importances_)
"""