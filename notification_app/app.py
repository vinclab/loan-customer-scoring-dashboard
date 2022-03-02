from flask import Flask, render_template, url_for, request
import os
from sqlalchemy import create_engine
import pandas as pd
from pickle import load

app = Flask(__name__)

#CONFIG________________________________________________________________________________________

# To generate a new secret key:
# >>> import os
# >>> os.urandom(24)
SECRET_KEY = '\xaff\x16\xf09$=\xee\x1d\xbcE\xfb\xe7+\x18~\x01\xf0z\xf9r\xf1\xd0r'

# columns_lst = ['SK_ID_CURR', 'PAYMENT_RATE', 'EXT_SOURCE_2', 'DAYS_BIRTH',
# 'EXT_SOURCE_3', 'AMT_ANNUITY', 'ANNUITY_INCOME_PERC', 'DAYS_ID_PUBLISH',
# 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'DAYS_REGISTRATION',
# 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'PREV_CNT_PAYMENT_MEAN',
# 'DAYS_LAST_PHONE_CHANGE', 'DAYS_EMPLOYED_PERC', 'BURO_DAYS_CREDIT_MAX',
# 'INSTAL_AMT_PAYMENT_SUM', 'INCOME_CREDIT_PERC',
# 'INSTAL_AMT_PAYMENT_MIN', 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN',
# 'CLOSED_DAYS_CREDIT_MAX', 'BURO_DAYS_CREDIT_ENDDATE_MAX',
# 'APPROVED_DAYS_DECISION_MAX', 'INSTAL_DBD_SUM', 'INSTAL_DBD_MAX',
# 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'BURO_AMT_CREDIT_SUM_MAX',
# 'INSTAL_AMT_PAYMENT_MAX', 'INSTAL_AMT_INSTALMENT_MAX',
# 'INSTAL_DAYS_ENTRY_PAYMENT_SUM', 'PREV_AMT_ANNUITY_MEAN']

#CONFIG________________________________________________________________________________________


#DATA________________________________________________________________________________________

# sqllite database connection
engine = create_engine('sqlite:///static\\tmp\\data_val.db').connect()

# table  will be returned as a dataframe.
df = pd.read_sql_table('data_val', engine)

# Reordering columns
#df = df[columns_lst]


#DATA________________________________________________________________________________________


#ALGO________________________________________________________________________________________

# load the model
model = load(open('static\\tmp\\pipeline_scoring.pkl', 'rb'))

#ALGO________________________________________________________________________________________


#UTILS________________________________________________________________________________________

def payment_rate(df_, number_):
    df_row = df_[df_['SK_ID_CURR']==number_]
    value =  df_row['PAYMENT_RATE'].iloc[0]

    return value

def row_data(df_, number_):
    df_row = df_[df_['SK_ID_CURR']==number_]

    return df_row.values

def prediction(model_, X_val_):
    # model prediction
    valid_probas = model_.predict_proba(X_val_)[:, 1]
    # Post prediction
    threshold=0.090
    y_pred = (valid_probas >= threshold).astype('int')

    return y_pred

#UTILS________________________________________________________________________________________


#VIEWS________________________________________________________________________________________

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result/')
def result():
    number = request.args.get('number')
    number = int(number)

    data = row_data(df, number)

    y_pred = prediction(model, data)

    value = payment_rate(df, number)

    return render_template('result.html',
                            prediction=y_pred,
                            id=number)

#VIEWS________________________________________________________________________________________

#RUN________________________________________________________________________________________

if __name__ == "__main__":
    app.run(debug=True)

#RUN________________________________________________________________________________________

