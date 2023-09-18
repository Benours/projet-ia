import joblib
from flask import Flask, render_template, request
from style import beers_style
import pandas as pd

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def formulaire():
    if request.method == "POST":
        # Récupérer les données soumises par le formulaire
        style = request.form["style"]
        style_id = beers_style.index(style) + 1
        size_l = request.form["size_l"]
        og = request.form["og"]
        boil_size = request.form["boil_size"]
        boil_time = request.form["boil_time"]
        boil_gravity = request.form["boil_gravity"]
        efficiency = request.form["efficiency"]

        # Faites quelque chose avec les données, par exemple les afficher dans la console

        data = {
            'Size(L)': [size_l],
            'OG': [og],
            'BoilSize': [boil_size],
            'BoilTime': [boil_time],
            'BoilGravity': [boil_gravity],
            'Efficiency': [efficiency],
            'OGPoly': [float(og) + float(og) ** 2],  # Vous avez mentionné 'OG2' dans votre question, nous allons utiliser OG^2 comme exemple
        }
  

        for i in range(1,177):
            if i == 164:
                continue
            elif i == 73:
                continue
            elif style_id == i:
                data[f'StyleID_{i}'] = True
            else:
                data[f'StyleID_{i}'] = False
            
       

        df = pd.DataFrame(data)

        # Load trained models
        ABV_model = joblib.load('models/random_forest_ABV.pkl')
        IBU_model = joblib.load('models/random_forest_IBU.pkl')

        # Test the models
        y_pred_ABV = ABV_model.predict(df)
        y_pred_IBU = IBU_model.predict(df)

        print("pred abv: ", y_pred_ABV)
        print("pred ibu: ", y_pred_IBU)

        # Retourner les données soumises dans le formulaire
        return render_template(
            "formulaire.html",
            style = style,
            size_l = size_l,
            og = og,
            boil_size = boil_size,
            boil_time = boil_time,
            boil_gravity = boil_gravity,
            efficiency = efficiency,
            beers_style = beers_style,
            y_pred_ABV = y_pred_ABV,
            y_pred_IBU = y_pred_IBU
        )

    # Si la méthode est GET ou si le formulaire n'a pas encore été soumis, afficher le formulaire vide
    return render_template(
        "formulaire.html",
        style="",
        size_l="",
        og="",
        boil_size="",
        boil_time="",
        boil_gravity="",
        efficiency="",
        beers_style=beers_style,
    )


if __name__ == "__main__":
    app.run()
