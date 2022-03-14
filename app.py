import dash
from dash import dcc, dash_table
from dash import html
from dash.dependencies import Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import requests
import time

# # Data reading with path information (val)
# df_val = pd.read_csv('C:\\Users\\vince\\OneDrive\\Documents\\data_scientist\\python_work\\projets\\07_loan_customer_scoring\\production\\savefig\\final_model\\cleaning\\df_val_cleaned.csv',sep=',')

SQLALCHEMY_DATABASE_URI = "postgres://uejnybrcakokbd:65ecd3f3834a74c2f5cef0469c38255c6c810416baff159d4441052d3a3b56dd@ec2-52-31-201-170.eu-west-1.compute.amazonaws.com:5432/d3ie9kgkbmr2fr"

# sqllite database connection
engine = create_engine(SQLALCHEMY_DATABASE_URI).connect()

# table  will be returned as a dataframe.
df_val = pd.read_sql_table('data_val'.lower(), engine)


id_lst = list(df_val['SK_ID_CURR'])
options=[{'label': i, 'value': i} for i in id_lst]

#Functions___________________________________________________________________________________________
def DisplayImagePIL(image, **kwargs):
    encoded_image = pil_to_b64(image, enc_format='png')

def pil_to_b64(im, enc_format="png", **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """

    buff = BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded

#Options___________________________________________________________________________________________

# columns number:
columns_number = df_val.shape[1]

# sample row for the initialisation of the table
df_sample = pd.DataFrame([np.zeros(columns_number)], columns=df_val.columns)
df_val = pd.concat([df_sample,df_val], ignore_index=True)

# graph dropdown list
graph_dropdown_lst = [x for x in df_val.columns if x not in ['SK_ID_CURR', 'CLASS']]

#creation of a temporary target/scores before the API call (in order not to block graph dropdowns)
df_val['SCORE'] = np.zeros(df_val.shape[0])
df_val['CLASS'] = np.ones(df_val.shape[0])

#API_______________________________________________________________________________________________
#prediction of all customers
r = requests.get('http://127.0.0.1:5000/notifications/')
data = r.json()
notifications = data['notifications']

notification_lst = [0] #sample value
classification_lst = [0] #sample value
score_lst = [0] #sample value
for notif in notifications:
    notification_lst.append(notif['SK_ID_CURR'])
    classification_lst.append(notif['classification'])
    score_lst.append(notif['score'])

threshold = 0.09 #to modify manually

#Application________________________________________________________________________________________
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH],
meta_tags=[{'name': 'viewport',
'content': 'width=device-width, initial-scale=1.0'}]
)
app.title = "Credit Notification"
server = app.server

colors = {
    'background': '#001440',
    'text': '#7FDBFF'
}

#Layout section: Bootstrap___________________________________________________________________________
app.layout = html.Div(style={'backgroundColor': colors['background'], 'color': colors['text']}, children=[dbc.Container([
    dbc.Row(
        [
            html.Hr(),
            html.Hr(),
            dbc.Col(
                    html.H1("NOTIFICATION DE CREDIT",
                        className='text-lg-center text-white'),
                        width='auto'),
            html.Hr(),
        ], justify="center"),

    dbc.Row(dbc.Col(
                    html.H5("Bienvenue dans l'outil de visualisation de données"),
                        className="text-sm-start",
                        width='auto'),
                ),

    dbc.Row(
        [
            
            dbc.Col([
                    dcc.Dropdown(id='id_dpdn', placeholder="Identifiant client",
                                options=options, className="text-lg-left text-dark"),
                    html.Div(id='id_dpdn_output')
                            ],
                    width='auto',
                    ),
            
            dbc.Col(
                    dbc.Button("OK", id="submit_button", className="me-2", n_clicks=0),
                    width={'size': 1, 'offset': 0, 'order': 1},
                    style={"margin-bottom": "20px"}
                    ),
            
        ]
        ),

    dbc.Row(
        dbc.Col(
                dbc.Spinner(dash_table.DataTable(
                    id='customer_table',
                    columns=[{"name": i, "id": i} for i in df_val.columns],
                    virtualization=True,
                    fixed_rows={'headers': True},
                    style_cell={'minWidth': 200, 'width': 300, 'maxWidth': 1000},
                    style_table={'height': 90},
                    style_header={
                    'backgroundColor': 'white',
                    'color': 'black'
                    },
                    style_data={
                        'backgroundColor': 'white',
                        'color': 'black'
                    },
                    ))
                )
            ),

    dbc.Row(dbc.Col(dbc.Card(
                dbc.CardBody(
                    [   html.Div(className="card-title1"),
                        dbc.Col(html.Div(id="class_result", className='text-md-center text-dark', style={'width': '100%', 'display': 'flex', 'align-items':'center', 'justify-content':'center'}), width='auto'),
                        html.Hr(),
                        dbc.Spinner(dcc.Graph(id="graph_score")),
                        dbc.Col(html.Div("Le score reflète la probabilité d'appartenance de classe (label 0: client à privilégier en zone verte; label 1: client à éviter en zone blanche, ex:259945), un seuil les séparant", className='text-sm-center text-dark',
                         style={'width': '100%', 'display': 'flex', 'align-items':'center', 'justify-content':'center'}), width='auto'),
                    ]
                ),
                className="attributes_card",
            ),
            className="attributes_card",
            width={'size': 6, 'offset': 0, 'order': 1}),
            justify="center",
            style={"margin-bottom": "20px"}),

    dbc.Row([
                    
            html.P("Valeurs de Shapley locales: ", className="text-lg-center text-white", style={'width': '100%', 'display': 'flex', 'align-items':'center', 'justify-content':'center'}),
            html.P("(veuiller sélectionner un client et patienter la mise à jour)", className="text-lg-center text-white", style={'width': '100%', 'display': 'flex', 'align-items':'center', 'justify-content':'center'}),
            dbc.Col(dcc.Loading(html.Iframe(
                            id="shap_graph",
                            src="",
                            style={"height": "125px", "width": "100%"},
                        )),
    )]),

    dbc.Row(dbc.Col(dbc.Card(
                dbc.CardBody(
                    [
                    html.Div("Valeurs de Shapley (globales et locales):", className="text-lg-center bg-warning text-white", style={'width': '100%', 'display': 'flex', 'align-items':'center', 'justify-content':'center'}),
                    html.Hr(),
                    html.A(id='shap_link_global', href='shap_link_global', className="text-lg-left", target="_blank"), 
                    html.Hr(),
                    html.A(id='shap_link_local', href='shap_link_local', className="text-lg-left", target="_blank"), 
                    ],              
                        )
                        )
                    ,className="attributes_card",
                     width="auto",
                     style={"margin-bottom": "20px"})
            , justify="center"),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                graph_dropdown_lst,
                'PAYMENT_RATE',
                id='crossfilter-xaxis-column',
                className="text-lg-center text-dark"
            ),
            dcc.RadioItems(['Client', 'Classe 0', 'Classe 1'],
             'Client',
                inline=True,
                id='classe',
                labelStyle={"margin": "auto", "padding": "5px"},
                labelClassName="text-lg-left text-white"
            )
        ],
        width={'size': 6, 'offset': 0, 'order': 1}),

        dbc.Col([
            dcc.Dropdown(
                graph_dropdown_lst,
                'ANNUITY_INCOME_PERC',
                id='crossfilter-yaxis-column',
                className="text-lg-center text-dark"
            ),
        ], width={'size': 6, 'offset': 0, 'order': 2})
    ], style={
        'padding': '10px 5px'
    }),

    dbc.Row([

            dbc.Col([
                    dbc.Spinner(dcc.Graph(id='distribution_var_1')),
                    ], style={'display': 'inline-block', 'width': '49%'}),

            dbc.Col([
                    dbc.Spinner(dcc.Graph(id='distribution_var_2')),
                    ], style={'display': 'inline-block', 'width': '49%'}),
            
            ]),

    dbc.Row([
        html.Hr(),
        html.Hr(),
        dbc.Col([
            dbc.Spinner(dcc.Graph(
                    id='crossfilter-indicator-scatter',
                    figure={'layout': {
                        'plot_bgcolor': 'white',
                        'paper_bgcolor': colors['background'],
                        'font': {
                            'color': 'black'
                        }}}))
                ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),
        html.Hr()
        ]),
        
    ],
    fluid=True)])


@app.callback(
    Output('shap_graph', 'src'),
    Input('submit_button', 'n_clicks'),
    State('id_dpdn', 'value'),)
def input_triggers_spinner(n_clicks,id):
    
    #local Shapley values
    if n_clicks==0:
        local_shap_graph_src = ""
    else:
        time.sleep(45)
        local_shap_graph_src = f'http://127.0.0.1:5000/notifications/interpretability/{id}'
    
    return local_shap_graph_src

# Table
@app.callback(
    
    Output('id_dpdn_output', 'children'),
    Output('customer_table', 'data'),
    Output('class_result', 'children'),
    Output('graph_score', 'figure'),
    #Output('shap_graph', 'src'),
    Output('shap_link_global', 'href'),
    Output('shap_link_global', 'children'),
    Output('shap_link_local', 'href'),
    Output('shap_link_local', 'children'),
    Input('submit_button', 'n_clicks'),
    State('id_dpdn', 'value'),
)
def customer_update(n_clicks, id):

    # Cutomer data
    if n_clicks==0:
        df_row = df_val[df_val['SK_ID_CURR']==0].round(2) # one customer example
    else:
        df_row = df_val[df_val['SK_ID_CURR']==id].round(2)
        # API call
        r = requests.get(f'http://127.0.0.1:5000/notifications/{id}')
        data = r.json()
        api_classif = data["classification"]
        api_score = data['score']

        df_row['SCORE'] = api_score
        df_row['CLASS'] = int(api_classif)

    # gauge 
    score = df_row['SCORE'].iloc[0]
    
    if n_clicks==0:
        score=0
        bar_color = 'green'
        result_sentence= "RESULTAT: "
    else:
        if score<threshold:
            bar_color = 'green'
            result_sentence= "CREDIT ACCORDÉ"
        else:
            bar_color = 'red'
            result_sentence= "CREDIT REFUSÉ"

    gauge = go.Figure(
        go.Indicator(
            domain={"x": [0, 1], "y": [0, 1]},
            value=score,
            title = {'text': "SCORE", 'font': {'size': 24}},
            delta = {'reference': threshold, 'decreasing': {'color': bar_color}, 'increasing': {'color': bar_color}},
            mode="gauge+number+delta",
            gauge={"axis": {"range": [None, 1]},
            "bar": {"color": bar_color},
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold},
            'steps': [
                {'range': [0, threshold], 'color': '#90EE90'},
                {'range': [threshold, 1], 'color': 'white'}],
            })
    )
    gauge.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font_color='black',
        font_size=15,
    )

    # text + local Shapley values
    if n_clicks==0:
        output = "Veuillez sélectionner puis cliquez sur OK"
        global_shap_link = ""
        global_shap_link_message = "Veuillez choisir un identifiant client"

        local_shap_link = ""
        local_shap_link_message = ""

        #local_shap_graph_src = ""
    else:
        output =  f'Vous avez sélectionné {id}'
        #shap_link = f'http://127.0.0.1:5000/notifications/interpretability/{id}'
        global_shap_link = 'http://127.0.0.1:5000/static/tmp/shap_global_feature_importance.png'
        global_shap_link_message = "Valeurs globales"

        local_shap_link = f"http://127.0.0.1:5000/notifications/interpretability/{id}"
        local_shap_link_message = "Valeurs locales (client)"

        #local_shap_graph_src = f'http://127.0.0.1:5000/notifications/interpretability/{id}'
        

    return output, df_row.to_dict('records'), result_sentence, gauge, global_shap_link, global_shap_link_message, local_shap_link, local_shap_link_message
    #return output, df_row.to_dict('records'), result_sentence, gauge, local_shap_graph_src, global_shap_link, global_shap_link_message, local_shap_link, local_shap_link_message

# graphs
@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Output('distribution_var_1', 'figure'),
    Output('distribution_var_2', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('classe', 'value'),
    Input('submit_button', 'n_clicks'),
    State('id_dpdn', 'value'),
    )
def comparison_update(xaxis_column_name, yaxis_column_name, classe, n_clicks, id):
    
    ## All customer data with API results
    dff = df_val.copy()
    dff['CLASS'] = classification_lst
    dff['SCORE'] = score_lst
    dff.drop(['SK_ID_CURR'], axis=1, inplace=True)

    ## Dataframe per class
    dff_class = dff.copy()
    if classe=='Client':
        dff_class = dff.copy()
        # Colors and sizes
        size_rule = dff_class['CLASS']==1
        dff_class['dummy_column_for_size'] = [0.1 if x==False else 2 for x in size_rule]

    if classe=='Classe 0':
        dff_class = dff.copy()
        dff_class = dff_class[dff_class['CLASS']==0]
        dff_class['dummy_column_for_size'] = 0.1

    if classe=='Classe 1':
        dff_class = dff.copy()
        dff_class = dff_class[dff_class['CLASS']==1]
        dff_class['dummy_column_for_size'] = 2

    # Scatter plot
    #Build the figure
    fig_scatter = px.scatter(dff_class[[xaxis_column_name,yaxis_column_name]],
            x=xaxis_column_name,
            y=yaxis_column_name,
            color=dff_class['SCORE'],
            color_continuous_scale=px.colors.sequential.Viridis,
            opacity=0.8,
            size=dff_class['dummy_column_for_size'],
            #hover_name=dff_class['SK_ID_CURR'],
            height=400,
            )

    # fig_scatter = go.Figure()

    # fig_scatter.add_trace(
    #             go.Scatter(
    #                 mode='markers',
    #                 x=dff_class[xaxis_column_name],
    #                 y=dff_class[yaxis_column_name],

    #                 marker=dict(
    #                 size=dff_class['dummy_column_for_size'],
    #                 cmax=39,
    #                 cmin=0,
    #                 color=dff_class['SCORE'],
    #                 opacity=0.8,
    #                 colorbar=dict(
    #                     title="Colorbar"
    #                 ),
    #                 colorscale="Viridis",
    #                 #height=400,
    #                 #hover_name=dff_class['SK_ID_CURR'],
    #             )))

    # axes config with customer position
    if n_clicks>0:
        df_row = df_val[df_val['SK_ID_CURR']==id]

        if classe=='Client':

            # Annotation of the customer
            x=df_row[xaxis_column_name].iloc[0]
            y=df_row[yaxis_column_name].iloc[0]
            fig_scatter.add_annotation(
                        #text="Client",
                        x=x, y=y, showarrow=True,
                        arrowhead = 1,
                        arrowwidth=2,
                        arrowsize =3,
                        #arrowside='start',
                        startstandoff=7,
                        arrowcolor= 'orange',
                        #bgcolor='white',
                        clicktoshow = False,
                        #height=15,
                        #width=40,
                        visible=True
                        ),
            
            x=df_row[xaxis_column_name]
            y=df_row[yaxis_column_name]
            # Add trace with large marker (class 1)
            fig_scatter.add_trace(
                go.Scatter(
                    mode='markers',
                    x=df_row[xaxis_column_name],
                    y=df_row[yaxis_column_name],
                    name="Classe 1",
                    marker=dict(
                        color='white',
                        size=15,
                        line=dict(
                            color='black',
                            width=1
                        )
                    ),
                    showlegend=True
                )
            ),

            # Add trace with large marker (class 0)
            fig_scatter.add_trace(
                go.Scatter(
                    mode='markers',
                    x=df_row[xaxis_column_name],
                    y=df_row[yaxis_column_name],
                    name="Classe 0",
                    marker=dict(
                        color='black',
                        size=2,
                        line=dict(
                            color='black',
                            width=1
                        )
                    ),
                    showlegend=True
                )
            ),

            # Add trace with other marker (customer)
            fig_scatter.add_trace(
                go.Scatter(
                    mode='markers',
                    marker_symbol='triangle-down',
                    x=df_row[xaxis_column_name],
                    y=df_row[yaxis_column_name],
                    name="Client",
                    marker=dict(
                        color='orange',
                        size=30,
                        line=dict(
                            color='orange',
                            width=1
                        )
                    ),
                    showlegend=True
                )
            ),

            # # Add trace with other marker (customer) #if go.scatter at first => too slow
            # fig_scatter.add_trace(
            #     go.Scatter(
            #         mode='markers',
            #         marker_symbol='cross',
            #         x=x,
            #         y=y,
            #         name="Client",
            #         marker=dict(
            #             color='orange',
            #             size=30,
            #             line=dict(
            #                 color='orange',
            #                 width=1
            #             )
            #         ),
            #         showlegend=True
            #     )
            # ),

            fig_scatter.update_layout(
                                    legend_title='Légende:',
                                    legend=dict(
                                        x=0,
                                        y=.9,
                                        itemsizing='trace',
                                        traceorder="normal",
                                        font=dict(
                                            #family="sans-serif",
                                            size=12,
                                            color="black"
                                        ),
                                    )
                                ),

    fig_scatter.update_layout(hovermode='closest',
                                coloraxis_colorbar=dict(
                                title="Score",
                                #tickvals=[6,7,8,9],
                                #ticktext=["1M", "10M", "100M", "1B"],
                                        )
                                )

    
    ## Distribution plots
    fig_distri_1 = px.histogram(dff_class, x=xaxis_column_name, height=400)

    fig_distri_2 = px.histogram(dff_class, x=yaxis_column_name, height=400)

    # axes config with customer position
    if n_clicks>0:
        df_row = df_val[df_val['SK_ID_CURR']==id]

        if classe=='Client':

            # Annotation of the customer
            x=df_row[xaxis_column_name].iloc[0]
            y=0
            fig_distri_1.add_annotation(
                        text="Client",
                        x=x, y=y, showarrow=True,
                        arrowhead = 1,
                        arrowwidth=2,
                        arrowsize =3,
                        #arrowside='start',
                        startstandoff=7,
                        arrowcolor= 'orange',
                        #bgcolor='white',
                        clicktoshow = False,
                        #height=15,
                        #width=40,
                        ),

            x=df_row[yaxis_column_name].iloc[0]
            y=0
            fig_distri_2.add_annotation(
                        text="Client",
                        x=x, y=y, showarrow=True,
                        arrowhead = 1,
                        arrowwidth=2,
                        arrowsize =3,
                        #arrowside='start',
                        startstandoff=7,
                        arrowcolor= 'orange',
                        #bgcolor='white',
                        clicktoshow = False,
                        #height=15,
                        #width=40,
                        ),

    return fig_scatter, fig_distri_1, fig_distri_2


    
if __name__=='__main__':
    app.run_server(debug=True, port=3000)




