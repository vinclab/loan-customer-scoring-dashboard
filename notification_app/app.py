from flask import Flask, render_template, request
import os
from sqlalchemy import create_engine
import pandas as pd
from pickle import load

from flask import jsonify
import json, requests

import shap
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)

#CONFIG_____________________________________________________________________________________

# database initialization
if os.environ.get('DATABASE_URL') is None:
    SQLALQUEMY_DATABASE_URI = 'sqlite:///static\\tmp\\data_val.db'
else:
    SQLALQUEMY_DATABASE_URI = os.environ['DATABASE_URL']


# columns (from feature selection)
columns_lst = ['SK_ID_CURR', 'PAYMENT_RATE', 'EXT_SOURCE_2', 'DAYS_BIRTH',
'EXT_SOURCE_3', 'AMT_ANNUITY', 'ANNUITY_INCOME_PERC', 'DAYS_ID_PUBLISH',
'AMT_GOODS_PRICE', 'AMT_CREDIT', 'DAYS_REGISTRATION',
'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'PREV_CNT_PAYMENT_MEAN',
'DAYS_LAST_PHONE_CHANGE', 'DAYS_EMPLOYED_PERC', 'BURO_DAYS_CREDIT_MAX',
'INSTAL_AMT_PAYMENT_SUM', 'INCOME_CREDIT_PERC',
'INSTAL_AMT_PAYMENT_MIN', 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN',
'CLOSED_DAYS_CREDIT_MAX', 'BURO_DAYS_CREDIT_ENDDATE_MAX',
'APPROVED_DAYS_DECISION_MAX', 'INSTAL_DBD_SUM', 'INSTAL_DBD_MAX',
'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'BURO_AMT_CREDIT_SUM_MAX',
'INSTAL_AMT_PAYMENT_MAX', 'INSTAL_AMT_INSTALMENT_MAX',
'INSTAL_DAYS_ENTRY_PAYMENT_SUM', 'PREV_AMT_ANNUITY_MEAN']
#___________________________________________________________________________________________


#DATA_______________________________________________________________________________________
# sqllite database connection
engine = create_engine(SQLALQUEMY_DATABASE_URI).connect()

# table  will be returned as a dataframe.
df = pd.read_sql_table('data_val', engine)

# Reordering columns
df = df[columns_lst]

#___________________________________________________________________________________________


#ALGO_______________________________________________________________________________________
# load the prediction model
pipeline = load(open('static\\tmp\\pipeline_scoring.pkl', 'rb'))

explainer = load(open('static\\tmp\\pipeline_explainer.pkl', 'rb'))

#___________________________________________________________________________________________


#UTILS______________________________________________________________________________________

# Convert fig to image
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

#Prediction__________________
def row_data(df_, number_list_):

    if len(number_list_)>1:
        df_row = df_[df_['SK_ID_CURR'].isin(number_list_)]
    else:
        id = number_list_[0]
        df_row = df_[df_['SK_ID_CURR']==id]

    return df_row

def prediction(pipeline_, df_row_):
    df_row_copy = df_row_.copy()
    # Matching the features
    df_row_copy.drop(['SK_ID_CURR'], axis=1, inplace=True)
    # model prediction
    valid_probas = pipeline_.predict_proba(df_row_copy)[:, 1]
    # Post prediction
    threshold=0.090
    y_pred = (valid_probas >= threshold).astype('int')

    return y_pred, valid_probas


# Interpretability_________
def sample(df_val_, id_):
    # Sample of val dataset
    df_val_sample = df_val_.sample(frac=0.01, replace=False, random_state=1)

    # check if the customer is in the sample
    check = df_val_sample['SK_ID_CURR']==id_
    check = check.value_counts()
    if len(check)==1:
        df_customer = df_val_[df_val_['SK_ID_CURR']==id_]
        
        df_val_sample = pd.concat([df_customer, df_val_sample])

    df_val_sample.set_index('SK_ID_CURR', inplace=True)

    return df_val_sample

def explain(pipeline_, explainer_, df_val_sample_):
    return explainer_.shap_values(pipeline_[:-1].transform(df_val_sample_.values))

def feature_importance(pipeline_, explainer_, df_val_, id_):
    
    #GLOBAL___
    # Average impact
    df_val_sample_ = sample(df_val_, id_)
    shap_values = explain(pipeline_, explainer_, df_val_sample_)

    # Average impact
    shap.summary_plot(shap_values, df_val_sample_, show=False)
    # fig
    fig = plt.gcf()
    fig.set_size_inches(70, 30)
    plt.title('Mean shap values', fontsize=50)
    plt.legend(loc=4, prop={'size': 40})
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=30)
    plt.xlabel('mean(|Mean SHAP|) (Average impact on model output magnitude)', fontsize=40)
    #fig = plt.gcf()
    img = fig2img(fig)
    img.save('static\\tmp\\shap_global_feature_importance.png')


    # LOCAL___
    index_lst = list(df_val_sample_.index)
    row = index_lst.index(id_)

    force_plot = shap.force_plot(explainer_.expected_value[1], explainer_.shap_values(df_val_sample_)[1][row,:], df_val_sample_.iloc[row,:], matplotlib=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    return  shap_html, img

#END POINTS______________________________________________________________________________________

# API call for prediction of one customer
@app.route('/notifications/<int:id>/')
def notification_one(id):
    customer_list = [id]
    data = row_data(df, customer_list)
    y_pred, probas = prediction(pipeline, data)

    probas = list(map(float, probas))
    probas = probas[0]
    probas = round(probas, 2)
    y_pred = list(map(float, y_pred))
    y_pred = y_pred[0]
    
    return jsonify({"SK_ID_CURR" : id, "classification" : y_pred, "score" : probas})

# API call for prediction of all customers listed
@app.route('/notifications/')
def notification_all():
    customer_list = list(df['SK_ID_CURR'])
    data = row_data(df, customer_list)
    y_pred, probas = prediction(pipeline, data)

    probas = list(map(float, probas))
    probas_lst = []
    for proba in probas:
        proba = round(proba, 2)
        probas_lst.append(proba)
    
    y_pred = list(map(float, y_pred))
    #y_pred = y_pred[0]

    notifications_lst = list()
    for k, v1, v2 in zip(customer_list, y_pred, probas_lst):
        notifications_lst.append({"SK_ID_CURR" : k, "classification" : v1, "score": v2})

    return jsonify({"notifications":notifications_lst})


# API call for interpretability
@app.route('/notifications/interpretability/<int:id>/')
def interpretability_one(id):


    shap_local_html, shap_global_img = feature_importance(pipeline, explainer, df, id)
    
    return shap_local_html

# API call for interpretability
@app.route('/notifications/interpretability/sample_with_id/<int:id>/')
def interpretability_sample(id):

    shap_local_html, shap_global_img = feature_importance(pipeline, explainer, df, id)

    return shap_global_img

#___________________________________________________________________________________________


#VIEWS______________________________________________________________________________________
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result/')
def result_one():
    if 'id' in request.args:
        id = int(request.args.get('id'))
        r = requests.get(f'http://127.0.0.1:5000/notifications/{id}')
        data = r.json()
        api_classif = data["classification"]

        if api_classif==1:
            description = "Le crédit est REFUSE"
        else:
            description = "Le crédit est ACCORDE"
            
        return render_template('result.html',
                                prediction=description,
                                customer_id=id)

    else:
        return "Error: No id field provided. Please specify an id."

@app.route('/result/all/')
def result_all():
    r = requests.get('http://127.0.0.1:5000/notifications/')
    data = r.json()
    notifications = data['notifications']

    return render_template('result_all.html', description=notifications)

@app.route('/shap/')
def result_shap():
    if 'id' in request.args:
        id = int(request.args.get('id'))
        shap_html = requests.get(f'http://127.0.0.1:5000/notifications/interpretability/{id}')

        return render_template('result_shap.html', shap_plot=shap_html)

#___________________________________________________________________________________________

#RUN________________________________________________________________________________________
if __name__ == "__main__":
    app.run(debug=True)
#___________________________________________________________________________________________

