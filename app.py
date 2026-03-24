from flask import Flask, render_template, request
import ast
import csv
import re
import pickle
from pathlib import Path
from unicodedata import normalize
from forms import ModelForm
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# to do : 
# find the average prediction for all models, including the full metrics group
# rather than use each model individually

app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecret"
BASE_DIR = Path(__file__).resolve().parent
PLAYER_MODELS_PATH = BASE_DIR / "player_models.csv"

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
MODEL_PARAMS = {
    "n_estimators": 50,
    "max_depth": None,
    "max_features": 0.2,
    "random_state": 42,
}
FEATURE_ALIASES = {
    "hard_hit_percent": "hh_percent",
}


def load_player_names():
    names = []
    seen = set()
    try:
        with PLAYER_MODELS_PATH.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = (row.get("Name") or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
    except FileNotFoundError:
        pass
    return sorted(names, key=str.casefold)


def parse_sequence(text):
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def load_model_features():
    model_features = {}
    try:
        with (BASE_DIR / "models.csv").open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                model_name = (row.get("model") or "").strip()
                if not model_name:
                    continue
                model_features[model_name] = parse_sequence(row.get("features", ""))
    except FileNotFoundError:
        pass
    return model_features


def slugify_player_name(player_name):
    normalized = normalize("NFKD", player_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.casefold()).strip("-")
    return slug


def load_player_data():
    names = []
    records = {}
    try:
        with PLAYER_MODELS_PATH.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = (row.get("Name") or "").strip()
                if not name or name.casefold() in records:
                    continue

                record = {
                    "name": name,
                    "fangraphs_id": (row.get("IDFANGRAPHS") or "").strip(),
                    "mlb_id": (row.get("MLBID") or "").strip(),
                    "models": parse_sequence(row.get("models", "")),
                }
                names.append(name)
                records[name.casefold()] = record
    except FileNotFoundError:
        pass

    return sorted(names, key=str.casefold), records


def load_training_rows():
    rows = []
    try:
        with (BASE_DIR / "df_ml.csv").open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass
    return rows


def build_model_training_data(feature_names, rows):
    x_values = []
    y_values = []
    for row in rows:
        try:
            year = int(float(row.get("year", "")))
            if year >= 2023:
                continue
            features = [float(row[feature]) for feature in feature_names]
            target = float(row["wRC+"])
        except (KeyError, TypeError, ValueError):
            continue
        x_values.append(features)
        y_values.append(target)
    return x_values, y_values


def train_models(model_features, rows):
    trained_models = {}
    for model_name, feature_names in model_features.items():
        x_values, y_values = build_model_training_data(feature_names, rows)
        if not x_values:
            continue
        model = RandomForestRegressor(**MODEL_PARAMS)
        model.fit(x_values, y_values)
        trained_models[model_name] = model
    return trained_models


def get_feature_value(model_form, feature_name):
    field_name = FEATURE_ALIASES.get(feature_name, feature_name)
    return float(getattr(model_form, field_name).data)


PLAYER_NAMES, PLAYER_RECORDS = load_player_data()
MODEL_FEATURES = load_model_features()
TRAINING_ROWS = load_training_rows()
TRAINED_MODELS = train_models(MODEL_FEATURES, TRAINING_ROWS)

@app.route("/")
def home():
    model_form = ModelForm()
    predictions = None
    player_name = (request.args.get("name") or "").strip()
    return render_template(
        "index.html",
        model_form=model_form,
        predictions=predictions,
        model_breakdown=[],
        player_names=PLAYER_NAMES,
        selected_player_name=player_name,
    )

@app.route("/predict", methods=["POST"])
def predict():
    model_form = ModelForm()
    predictions = None
    player_name = (request.form.get("player_name") or "").strip()

    model_breakdown = []
    if model_form.validate_on_submit():
        try:
            player_record = PLAYER_RECORDS.get(player_name.casefold()) if player_name else None
            if player_record and player_record.get("models"):
                for model_name in player_record["models"]:
                    model = TRAINED_MODELS.get(model_name)
                    feature_names = MODEL_FEATURES.get(model_name, [])
                    if not model or not feature_names:
                        continue
                    feature_values = [get_feature_value(model_form, feature_name) for feature_name in feature_names]
                    model_breakdown.append((model_name, model.predict([feature_values])[0]))
                if model_breakdown:
                    predictions = round(np.mean([value for _, value in model_breakdown]))

            if predictions is None:
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

    return render_template(
        "index.html",
        model_form=model_form,
        predictions=predictions,
        model_breakdown=[(name, round(value)) for name, value in model_breakdown],
        player_names=PLAYER_NAMES,
        selected_player_name=player_name,
    )

@app.route("/original")
def original():
    return render_template("original.html", predictions=None, form_values={})

@app.route("/original-docs")
def original_docs():
    return render_template("original_docs.html")


@app.route("/player-links")
def player_links():
    player_name = (request.args.get("name") or "").strip()
    if not player_name:
        return {"baseball_savant": "", "fangraphs": ""}

    record = PLAYER_RECORDS.get(player_name.casefold())
    if not record:
        return {"baseball_savant": "", "fangraphs": ""}

    slug = slugify_player_name(record["name"])
    baseball_savant = f"https://baseballsavant.mlb.com/savant-player/{slug}-{record['mlb_id']}"
    fangraphs = f"https://www.fangraphs.com/players/{slug}/{record['fangraphs_id']}/stats/batting"
    return {"baseball_savant": baseball_savant, "fangraphs": fangraphs}


@app.route("/documentation")
def documentation():
    player_name = (request.args.get("name") or "").strip()
    record = PLAYER_RECORDS.get(player_name.casefold()) if player_name else None
    model_details = []
    if record:
        slug = slugify_player_name(record["name"])
        player_links = {
            "baseball_savant": f"https://baseballsavant.mlb.com/savant-player/{slug}-{record['mlb_id']}",
            "fangraphs": f"https://www.fangraphs.com/players/{slug}/{record['fangraphs_id']}/stats/batting",
        }
        for model_name in record.get("models", []):
            model_details.append({
                "name": model_name,
                "features": MODEL_FEATURES.get(model_name, []),
            })
    else:
        player_links = {"baseball_savant": "", "fangraphs": ""}

    return render_template(
        "documentation.html",
        selected_player_name=record["name"] if record else "",
        selected_player_links=player_links,
        selected_player_models=model_details,
        player_names=PLAYER_NAMES,
    )

if __name__ == '__main__':
    app.run(debug=True)
