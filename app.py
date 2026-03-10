from flask import Flask, render_template, request
import pickle
from forms import ModelForm
import numpy as np

# to do : 
# find the average prediction for all models, including the full metrics group
# rather than use each model individually

app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecret"

LEGACY_FIELDS = [
    "exit_velocity",
    "brl_percent",
    "hh_percent",
    "chase_percent",
    "whiff_percent",
    "k_percent",
    "bb_percent",
]

model_1=pickle.load(open("model_1.sav","rb"))
model_2=pickle.load(open("model_2.sav","rb"))
model_3=pickle.load(open("model_3.sav","rb"))
model_4=pickle.load(open("model_4.sav","rb"))
model_5=pickle.load(open("model_5.sav","rb"))
model_all=pickle.load(open("model_all.sav","rb"))

@app.route("/")
def home():
    model_form = ModelForm()
    predictions = None
    return render_template("index.html", model_form=model_form, predictions=predictions, model_breakdown=[])

@app.route("/predict", methods=["POST"])
def predict():
    model_form = ModelForm()
    predictions = None

    model_breakdown = []
    if model_form.validate_on_submit():
        try:
            brl_percent = float(model_form.brl_percent.data)
            k_percent = float(model_form.k_percent.data)
            bb_percent = float(model_form.bb_percent.data)
            exit_velocity = float(model_form.exit_velocity.data)
            chase_percent = float(model_form.chase_percent.data)
            whiff_percent = float(model_form.whiff_percent.data)
            hh_percent = float(model_form.hh_percent.data)

            model1 = [brl_percent, exit_velocity, k_percent, bb_percent, chase_percent]
            model2 = [brl_percent, k_percent, bb_percent]
            model3 = [brl_percent, exit_velocity, k_percent, bb_percent]
            model4 = [brl_percent, exit_velocity, hh_percent, k_percent, bb_percent, chase_percent]
            model5 = [brl_percent, k_percent, bb_percent, chase_percent]
            modelall = [brl_percent, exit_velocity, hh_percent, k_percent, bb_percent, whiff_percent, chase_percent]

            model_breakdown = [
                ("model_all", model_all.predict([modelall])[0]),
                ("model_1", model_1.predict([model1])[0]),
                ("model_2", model_2.predict([model2])[0]),
                ("model_3", model_3.predict([model3])[0]),
                ("model_4", model_4.predict([model4])[0]),
                ("model_5", model_5.predict([model5])[0]),
            ]
            predictions = round(np.mean([value for _, value in model_breakdown]))
        except (ValueError, TypeError):
            predictions = None
            model_breakdown = []

    origin_template = request.form.get("origin_template", "modern")
    if origin_template == "original":
        form_values = {name: request.form.get(name, "") for name in LEGACY_FIELDS}
        return render_template("original.html", predictions=predictions, form_values=form_values)

    return render_template("index.html", model_form=model_form, predictions=predictions, model_breakdown=[(name, round(value)) for name, value in model_breakdown])

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

@app.route("/original")
def original():
    return render_template("original.html", predictions=None, form_values={})

@app.route("/original-docs")
def original_docs():
    return render_template("original_docs.html")

if __name__ == '__main__':
    app.run(debug=True)
