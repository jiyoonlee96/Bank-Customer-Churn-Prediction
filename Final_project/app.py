from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import plotly
import plotly.express as px
from models import knn_model, svm_model


app = Flask(__name__)

df = pd.read_csv("static/Bank_Customer_Churn_Prediction.csv")

@app.route("/")

def select_model():
    return render_template('Analysis.html')

@app.route('/analyze_knn', methods=["POST"])
def knn():
    usertext = request.form["usertext"]
    k = int(usertext)
    accuracy, cm_df = knn_model(df, k)
    fig = px.imshow(cm_df,
                    text_auto=True,
                    color_continuous_scale='agsunset',
                    labels=dict(x="Predicted label ", y="True label"),
                    x=['Not Churn', 'Churn'],
                    y=['Not Churn', 'Churn']
                )
    fig.update_layout(
    title=f'Confusion matrix for K = {k} :',
    width=800,
    height=800,
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('analyze_knn.html', accuracy=accuracy, graphJSON=graphJSON)

@app.route('/analyze_svm', methods=["POST"])
def svm():
    usertext = request.form["usertext"]
    reg = int(usertext)
    accuracy, cm_df = svm_model(df, reg)
    fig = px.imshow(cm_df,
                text_auto=True,
                color_continuous_scale='agsunset',
                labels=dict(x="Predicted label ", y="True label"),
                x=['Not Churn', 'Churn'],
                y=['Not Churn', 'Churn']
               )
    fig.update_layout(
    title=f'Confusion matrix for C = {reg} :',
    width=800,
    height=800,
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('analyze_svm.html', accuracy=accuracy, graphJSON=graphJSON)


if __name__ == "__main__":
    app.run(debug=True)