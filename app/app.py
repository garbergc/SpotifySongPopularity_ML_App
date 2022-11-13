from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from pathlib import Path
import uuid


app = Flask(__name__)

#@app.route("/")
@app.route("/", methods=["GET", "POST"])

def hello_world():
    request_type_str = request.method
    if request_type_str == "GET":
        path = "static/baseimage.jpg"
        return render_template("index.html", href=path)
    else:
        text = request.form["text"]
        random_string = uuid.uuid4().hex
        path = "static/"+random_string+".jpg"

        # load model
        np_arr = floatsome_to_np_array(text).reshape(1, -1)
        pkl_filename="TrainedModel/StackedPickle_Regression.pkl"
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        output = plot_graphs(pickle_model,new_input_arr=np_arr,output_file=path)
        return render_template("index.html", href=path, output=output)

# plotting graphs for application
def plot_graphs(model, new_input_arr, output_file):
    filename = 'data/final_dataframe_regression.csv'
    df = pd.read_csv(filename)
    
    # creating subplots in loop
    plt.figure(figsize=(15, 9))

    features = df.columns[:-1]
    target = df['popularity']
    
    new_preds = model.predict(new_input_arr)

    for i, col in enumerate(features):
        
        feature_input = np.array(new_input_arr[0][i])
        
        plt.subplot(3, 5, i+1)
        x = df[col]
        y = target
        plt.scatter(x, y, marker='o')
        plt.plot(feature_input, new_preds, marker='*', color='yellow', ms=20)
        #plt.title("Variation In Song Popularity",loc="Left")
        plt.xlabel(col)
        plt.ylabel('popularity')
        plt.tight_layout()
   
    plt.suptitle('Variation In Song Popularity')
    plt.subplots_adjust(top=0.90)
    plt.savefig(output_file)
    #plt.show()

    new_preds_out = round(new_preds[0], 2)
    return new_preds_out

def floatsome_to_np_array(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)