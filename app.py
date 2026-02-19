from flask import Flask, render_template, request
import pickle
from forms import ModelForm
import numpy as np

# to do : 
# find the average prediction for all models, including the full metrics group
# rather than use each model individually

app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecret"

model_1=pickle.load(open("model_1.sav","rb"))
model_2=pickle.load(open("model_2.sav","rb"))
model_3=pickle.load(open("model_3.sav","rb"))
model_4=pickle.load(open("model_4.sav","rb"))
model_all=pickle.load(open("model_all.sav","rb"))

@app.route("/")
def home():
    predictions = ''
    return render_template("index.html",**locals())

@app.route("/predict", methods=["GET", "POST"])
def predict():
    model_form = ModelForm()

    brl_percent = float(int(model_form.brl_percent.data))
    k_percent = float(int(model_form.k_percent.data))
    bb_percent = float(int(model_form.bb_percent.data))
    exit_velocity = float(int(model_form.exit_velocity.data))
    chase_percent = float(int(model_form.chase_percent.data))
    whiff_percent = float(int(model_form.whiff_percent.data))
    hh_percent = float(int(model_form.hh_percent.data))

    predictions = round(np.mean([model_all.predict([[brl_percent, exit_velocity, hh_percent, k_percent, bb_percent, whiff_percent, chase_percent]])[0],model_1.predict([[brl_percent, k_percent, bb_percent]])[0],model_2.predict([[brl_percent, exit_velocity, k_percent, bb_percent]])[0], model_3.predict([[brl_percent, k_percent, bb_percent, chase_percent]])[0],model_4.predict([[brl_percent, exit_velocity, k_percent, bb_percent, whiff_percent]])[0]]))

    return render_template("index.html", **locals(), form=model_form)

@app.route("/models")
def models():
    return render_template("models.html", **locals())

if __name__ == '__main__':
    app.run(debug=True)