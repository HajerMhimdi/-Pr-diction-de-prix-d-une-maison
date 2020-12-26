import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import pandas as pd

pipelineM= joblib.load(open('dataPreparation.pkl', 'rb'))
predictModel = joblib.load(open('HistGradientBoostingRegressor.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/house')
def house():
    return render_template('prixMaison.html')

@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form.get('longitude')
    data2 = request.form.get('latitude')
    data3 = request.form.get('housing_median_age')
    data4 = request.form.get('total_rooms')
    data5 = request.form.get('total_bedrooms')
    data6 = request.form.get('population')
    data7 = request.form.get('households')
    data8 = request.form.get('median_income')
    data9 = request.form.get('ocean_proximity')
 #here the Feature-extraction and the total bedrooms was droped so we will not put it in the array
    rooms_per_households= float(data4)/ float(data7)
    bedrooms_per_room = float(data5) / float(data4)
    population_per_household = float(data6) / float(data7)

    arr = np.array([data1, data2, data3, data4, data5,data6, data7, data8, data9, rooms_per_households, bedrooms_per_room,
     population_per_household])

    data_ft = pd.DataFrame(data=[arr], columns=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms',
    'population','households','median_income','ocean_proximity','rooms_per_household','bedrooms_per_room',
    'population_per_household'])

    pred = pipelineM.transform(data_ft)
    predictdata= predictModel.predict(pred)
    return render_template('prixMaison.html' , prixMaison ='{}'.format(predictdata))

if __name__ == '__main__':
    app.run(debug=True)