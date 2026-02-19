from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, RadioField
from wtforms.validators import DataRequired

class ModelForm(FlaskForm):
    brl_percent = StringField("Barrel %",validators=[DataRequired()])
    k_percent = StringField("K %",validators=[DataRequired()])
    bb_percent = StringField("BB %",validators=[DataRequired()])
    exit_velocity = StringField("Exit Velocity",validators=[DataRequired()])
    chase_percent = StringField("Chase %",validators=[DataRequired()])
    whiff_percent = StringField("Whiff %",validators=[DataRequired()])
    hh_percent = StringField("Hard Hit %",validators=[DataRequired()])

    submit = SubmitField("Predict wrC+")