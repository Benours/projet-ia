from flask import Flask, render_template, request
from style import beers_style

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def formulaire():
    if request.method == "POST":
        # Récupérer les données soumises par le formulaire
        style = request.form["style"]
        style_index = beers_style.index(style)
        size_l = request.form["size_l"]
        og = request.form["og"]
        boil_size = request.form["boil_size"]
        boil_time = request.form["boil_time"]
        boil_gravity = request.form["boil_gravity"]
        efficiency = request.form["efficiency"]

        # Faites quelque chose avec les données, par exemple les afficher dans la console
        print(
            f"Style_id : {style_index+1}, Size(L): {size_l}, OG: {og}, BoilSize: {boil_size}, BoilTime: {boil_time}, BoilGravity: {boil_gravity}, Efficiency: {efficiency}"
        )

        # Retourner les données soumises dans le formulaire
        return render_template(
            "formulaire.html",
            style=style,
            size_l=size_l,
            og=og,
            boil_size=boil_size,
            boil_time=boil_time,
            boil_gravity=boil_gravity,
            efficiency=efficiency,
            beers_style=beers_style,
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
