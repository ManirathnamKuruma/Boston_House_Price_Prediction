from flask import Flask, render_template, request
import jsonify
import requests
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)
# load model
model = load_model('model.h5')

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        crim = float(request.form['crim'])
        zn = float(request.form['zn'])
        indus = float(request.form['indus'])
        chas = float(request.form['chas'])
        nox = float(request.form['nox'])
        rm = float(request.form['rm'])
        age = float(request.form['age'])
        dis = float(request.form['dis'])
        rad = float(request.form['rad'])
        tax = float(request.form['tax'])
        ptratio = float(request.form['ptratio'])
        blck = float(request.form['black'])
        lstat = float(request.form['lstat'])
        prediction = model.predict(keras.utils.normalize([[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,blck,lstat]]))
        output=prediction.item()
        return render_template('index.html',prediction_text="Median Value of Owner Occupied Homes in terms of Dollars: {}".format(round(output,2)))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

