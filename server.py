import os
import pickle
import datetime

import numpy as np

import plotly
import plotly.subplots
import plotly.graph_objects as go
from shapely.geometry.polygon import Point
from shapely.geometry.polygon import Polygon

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, flash
from flask import render_template, redirect

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired, ValidationError
from wtforms import validators
from wtforms import StringField, SubmitField, FileField, IntegerField, DecimalField, SelectField

from utils import polygon_random_point

from ensembles import RandomForestMSE, GradientBoostingMSE
import pandas as pd
from jinja2 import FileSystemLoader, Environment


app = Flask(__name__, template_folder='server_html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)
models = []
current = 1


class FileForm(FlaskForm):
    file_path = FileField('Тестовая выборка', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Загрузить')
    

class TextForm(FlaskForm):
    text = StringField('Text', validators=[DataRequired()])
    def validate_text(form, field):
        if field.data != 'RandomForest' and field.data != 'GradientBoosting':
            raise ValidationError("You should write 'RandomForest' or 'GradientBoosting'!")
    submit = SubmitField('Get Result')

class NumberForm(FlaskForm):
    n = IntegerField('number')
    def validate_n(form, field):
        if field.data <= 0:
            raise ValidationError("Only positive numbers!")


@app.route('/')
@app.route('/start')
def index():
    return render_template('start.html')


class ModelForm(FlaskForm):   
    model_type = SelectField("Модель", choices=["RandomForest", "GradientBoosting"], validators=[DataRequired()])
    n_estimators = IntegerField('n_estimators', validators=[DataRequired(), validators.NumberRange(min=1, max=2000)])
    feature_subsample_size = IntegerField('feature_subsample_size (0 if None)', validators=[validators.NumberRange(min=0, max=500)])
    max_depth = IntegerField('max_depth (0 if None)', validators=[validators.NumberRange(min=0, max=100)])
    learning_rate = DecimalField('learning_rate (>= 0, <= 1; 0 if RandomForest)', validators=[validators.NumberRange(min=0, max=1)])
    target = StringField('Целевая переменная', validators=[DataRequired()])
    train_path = FileField('Обучающая выборка', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    val = SelectField("Хочу смотреть результаты на валидационной выборке", choices=["Да", "Нет"], validators=[DataRequired()])
    val_path = FileField('Валидационная выборка (если в предыдущем вопросе ответ "Нет", прикрепите обучающую)', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Создать модель')


class InfoForm(FlaskForm):
    model_number = IntegerField('Номер модели', validators=[validators.NumberRange(min=1, max=len(models) + 1)])
    action = SelectField("Что Вы хотите сделать", choices=["Посмотреть сведения о модели", "Получить предсказание"], validators=[DataRequired()])
    submit = SubmitField('Отправить запрос')


@app.route('/models', methods=['GET', 'POST'])
def prepare_model():
    cur_model = ModelForm()
    error = False
    if request.method == 'POST' and cur_model.validate_on_submit():
        X = pd.read_csv(cur_model.train_path.data)
        if cur_model.target.data not in X.columns:
            flash("Неправильное имя целевой переменной")
            error = True
        elif cur_model.feature_subsample_size.data > (X.shape[1] - 1):
            flash("feature_subsample_size больше числа признаков")
            error = True
        elif cur_model.model_type.data == 'GradientBoostingMSE' and cur_model.learning_rate.data == 0:
            flash("learning_rate должен быть ненулевым")
            error = True
        if cur_model.val.data == 'Да':
            data_val = pd.read_csv(cur_model.val_path.data)
            if X.shape != data_val.shape:
                flash("Размеры обучающей и валидационной выборок не совпадают")
                error = True
        
        if not error:
            X_train = X.drop(cur_model.target.data, axis=1)
            X_train = X_train.to_numpy()
            y_train = X[cur_model.target.data]
            y_train = y_train.to_numpy()
            if cur_model.val.data == 'Да':
                X_val = data_val.drop(cur_model.target.data, axis=1)
                X_val = X_val.to_numpy()
                y_val = data_val[cur_model.target.data]
                y_val = y_val.to_numpy()
            else:
                X_val = None
                y_val = None
            if cur_model.feature_subsample_size.data == 0:
                fss = None
            if cur_model.max_depth.data == 0:
                max_depth = None
            if cur_model.model_type.data == 'RandomForest':
                model = RandomForestMSE(cur_model.n_estimators.data, max_depth, fss)
            else:
                model = GradientBoostingMSE(cur_model.n_estimators.data, float(cur_model.learning_rate.data), max_depth, fss)
            model.fit(X_train, y_train, X_val, y_val)
            models.append((cur_model.model_type.data, model, len(models) + 1, cur_model.val_path.data.filename))
        return redirect(url_for('prepare_model'))
    return render_template('from_form.html', form=cur_model)


@app.route('/list_of_models', methods=['GET'])
def information():
    return render_template('templ.html', q=models)


@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    test_form = FileForm()
    if request.method == 'POST' and test_form.validate_on_submit():
        X_test = pd.read_csv(test_form.file_path.data)
        X_test = X_test.to_numpy()
        pred = models[current - 1][1].predict(X_test)
        return render_template('result.html', pred=pred, l=list(range(len(pred))))
    return render_template('predict.html', form=test_form)


@app.route('/models_inf', methods=['GET', 'POST'])
def model_information():
    info = InfoForm()
    if request.method == 'POST' and info.validate_on_submit():
        if info.action.data == 'Посмотреть сведения о модели' and len(models[info.model_number.data - 1][1].rmse_test) != 0:
            return render_template('info.html', l_of_models=models, number=info.model_number.data, l=list(range(len(models[info.model_number.data - 1][1].rmse))))
        elif info.action.data == 'Посмотреть сведения о модели':
            return render_template('info_without_val.html', l_of_models=models, number=info.model_number.data, l=list(range(len(models[info.model_number.data - 1][1].rmse))))
        elif info.action.data == "Получить предсказание":
            current = info.model_number.data
            return redirect(url_for('prediction'))
    return render_template('models_inf.html', form=info)





    
