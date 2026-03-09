from flask_wtf import FlaskForm
from wtforms import DecimalField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class ModelForm(FlaskForm):
    number_validators = [DataRequired(), NumberRange(min=0, max=100)]

    brl_percent = DecimalField("Barrel %", validators=number_validators)
    k_percent = DecimalField("K %", validators=number_validators)
    bb_percent = DecimalField("BB %", validators=number_validators)
    exit_velocity = DecimalField("Exit Velocity", validators=number_validators)
    chase_percent = DecimalField("Chase %", validators=number_validators)
    whiff_percent = DecimalField("Whiff %", validators=number_validators)
    hh_percent = DecimalField("Hard Hit %", validators=number_validators)

    submit = SubmitField("Predict wrC+")
