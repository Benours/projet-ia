import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset and preprocess it
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./beers/recipeData.csv", encoding="latin1")

# Drop unnecessary columns
columns_to_drop = ["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
                   "PrimaryTemp", "SugarScale", "FG", "Color", "BrewMethod"]
df = df.drop(columns=columns_to_drop)
df.dropna(inplace=True)

# Filter data
df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
df = df[(df["IBU"] <= 150) & (df["IBU"] > 0)]
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]
df['OGPoly'] = df["OG"] + df['OG'] ** 2

# Encode categorical variables
new_len = len(df)
one_hot = pd.get_dummies(df[["StyleID"]], columns=["StyleID"])
df = df.drop(columns=["StyleID"])
df = df.join(one_hot)

# Prepare data for ABV and IBU prediction
df_X = df.drop(columns=["ABV", "IBU"])
df_X_IBU = df.drop(columns=['ABV', 'IBU'])
df_y_ABV = df["ABV"]
df_y_IBU = df["IBU"]

# Split data into train and test sets
X_train_ABV, X_test_ABV, y_train_ABV, y_test_ABV = train_test_split(df_X, df_y_ABV)
X_train_IBU, X_test_IBU, y_train_IBU, y_test_IBU = train_test_split(df_X_IBU, df_y_IBU)

# Load trained models
ABV_model = joblib.load('models/random_forest_ABV.pkl')
IBU_model = joblib.load('models/random_forest_IBU.pkl')


# Créez un dictionnaire de données avec les valeurs de test
data = {
    'Size(L)': [21.77],
    'OG': [1.055],
    'BoilSize': [28.39],
    'BoilTime': [75],
    'BoilGravity': [1.038],
    'Efficiency': [70],
    'OGPoly': [1.055 + 1.055 ** 2],  # Vous avez mentionné 'OG2' dans votre question, nous allons utiliser OG^2 comme exemple
}

for i in range(1, 177):  # Vous avez mentionné 176 StyleID
    if i == 164:
        continue
    elif i == 73:
        continue
    else:
        data[f'StyleID_{i}'] = False

# Créez un DataFrame à partir du dictionnaire
test_data = pd.DataFrame(data)

# Test the models
y_pred_ABV = ABV_model.predict(test_data)
y_pred_IBU = IBU_model.predict(X_test_IBU)

print("Resultat ABV :", y_pred_ABV)
print("Resultat IBU :", y_pred_IBU)
