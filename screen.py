import pandas as pd
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./beers/recipeData.csv", encoding="latin1")

df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale", "FG", "Color"])

df.dropna(inplace=True)

df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
df = df[(df["IBU"] <= 150) & (df["IBU"] > 0)]  # IBU max == 150 selon wikipedia
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]

one_hot = pd.get_dummies(df[["StyleID"]], columns=["StyleID"])
df = df.join(one_hot)
df = df.drop(
    columns=["StyleID"]
)

fig, axes = plt.subplots(5, 5)
plt.subplots_adjust(wspace=10, hspace=10)
for ax, style in zip(axes.flatten(), one_hot.columns):
    df.hist(column="IBU", bins=50, ax=ax, facecolor='#00FF00', density=True)
    df[df[style]].hist(column="IBU", bins=50, ax=ax, facecolor='#FF000075', density=True)
    ax.set_title(style)

# Afficher le plot
plt.show()

plt.show()