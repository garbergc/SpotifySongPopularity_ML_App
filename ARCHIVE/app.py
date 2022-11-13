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
from matplotlib import pyplot
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
        path = "static/"+random_string+".html"

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
    fig = make_subplots(
    rows=3, cols=5
    )

    features = df.columns[:-1]
    target = df['popularity']
    
    col_num = 1
    row_num = 1
    
    new_preds = model.predict(new_input_arr)

    for i, col in enumerate(features):
        
        feature_input = np.array(new_input_arr[0][i])
        
        if i < 5:
            row_num = 1
            col_num = i+1
        elif i > 4 and i < 10:
            row_num = 2
            col_num = i-4
        elif i > 9:
            row_num = 3
            col_num = i-9
        
        fig.add_trace(
            go.Scatter(x=df[col],y=target,mode='markers',
            marker=dict(
                    color="#003366"),
                line=dict(color="#003366",width=1)),
            row=row_num, col=col_num
        )

        fig.add_trace(
        go.Scatter(
            x=feature_input,
            y=new_preds,
            mode='markers',
            marker=dict(
                color="#FFCC00",size=15),
            line=dict(color="#FFCC00",width=1)),
            row=row_num, col=col_num
        )

        # Update xaxis properties
        fig.update_xaxes(title_text=str(col), row=row_num, col=col_num)

        # Update yaxis properties
        fig.update_yaxes(title_text="popularity", row=row_num, col=col_num)

    # Update title and height
    fig.update_layout(height=600, width=1400, title_text="Variation In Song Popularity", showlegend=False)
    #fig.write_image(output_file,width=1200,engine="kaleido")
    with Path(output_file).open("w") as f:
        f.write(fig.to_html())
    #fig.show()
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