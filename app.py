import json
import pickle
from flask import Flask, request, app, url_for, render_template, jsonify
import numpy as np 
import pandas as pd 

app = Flask(__name__)

model = pickle.load(open('regmodel.pkl', 'rb'))
scale = pickle.load(open('scalling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scale.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output)
    
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)