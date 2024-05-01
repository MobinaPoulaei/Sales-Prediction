# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:17:24 2024

@author: Mobina
"""

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open ('random_forest_model.pkl', 'rb') as file:
    rf = pickle.load(file)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [int(x) for x in request.form.values()]
    prediction = rf.predict([data])[0]
    return render_template('index.html', prediction_text=f'Predicted Sales: {prediction:.2f}')

if __name__=='__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)