import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask,render_template,request,app,Response
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application
scaler = pickle.load(open("Models/standardScalar.pkl",'rb'))
model = pickle.load(open("Models/modelForPrediction.pkl",'rb'))

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predictdiabetes",methods=["GET","POST"])
def predict_data():
    if request.method == "POST":
        Pregnancies = request.form.get('Pregnancies')
        Glucose = request.form.get('Glucose')
        BloodPressure = request.form.get('BloodPressure')
        SkinThickness =request.form.get('SkinThickness')
        Insulin = request.form.get('Insulin')
        BMI = request.form.get('BMI')
        DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
        Age = request.form.get('Age')
        new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI,DiabetesPedigreeFunction,Age]])
        prediction = model.predict(new_data)

        if prediction[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Not Diabetic'
        return render_template('singlepredictio.html',result=result)

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
