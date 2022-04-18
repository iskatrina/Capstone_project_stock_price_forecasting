from datetime import datetime
from datetime import date,timedelta
import pandas as pd
import dash
from dash import dash_table
from dash import dcc
from dash import html
import plotly.graph_objs as go
import joblib

from utilities.supportive_functions import preprocess, calculate_results

from dash.dependencies import Input, Output, State

# ---------------------------------------

app = dash.Dash(__name__)

#reading data
df = preprocess(pd.read_csv(f'../data/ticker_data/AAPL_full_data.csv'))
df = df.iloc[:,:6]
data = df.copy()
data.date = pd.to_datetime(data.date)
data = data.set_index('date')

index_split = int(len(data['adj close']) * 0.8), int(len(data['adj close']) * 0.2), int(len(data['adj close']) )
df_train = data[:index_split[0]]
df_test = data[index_split[0]:]
df_train_x = df_train[['open','close','high','low']]
df_train_y = df_train['adj close']


layout = go.Layout(
yaxis=dict( domain=[1, 1]),
legend=dict(traceorder='reversed'),
yaxis2=dict(domain=[1, 1]),
yaxis3=dict(domain=[1, 1])
)

fig = go.Figure(layout = layout)

fig.add_trace(go.Scatter(y=data['adj close'][:2111], x=data.index[:2111], name='training data'))
fig.add_trace(go.Scatter(y=data['adj close'][2111:], x=data.index[2111:], name='test data'))

fig.update_layout(title='APPLE Price- adjusted close ($)', xaxis_title="date",
                  yaxis_title="Stock price($)",)

###########################################################################

###########################################################################

all_options = {
    'LSTM': ['Deep Learning LSTM'],
    'SARIMA': ['SARIMA'],
    'Random Forest Regression': ['Random Forest'],
    'XGBOOST Regression': ['XGBoost'],
    'Q-learning': ['Q-learning']
            }


portfolio = {
    'VZ' : ['VZ'],
    'TSLA' : ['Tesla'],
    'INTC' : ['INTC'],
    'CAT'  : ['CAT'],
    'JNJ'  : ['JNJ'],
    'PFE'  : ['PFE'],
    'AAPL' : ['AAPL'],
    'MSFT' : ['MSFT']
}


app.layout = html.Div([
                html.H1('Stock Price Prediction Dashboard'),

                html.Div([
                html.H3('Choose a model for prediction:',style = {'paddingRight':'30px'}),
                    dcc.Dropdown(
                        id='models-dropdown',
                        options=all_options,
                        value=['LSTM'],
                        multi=False
                    )
                ], style = {'display':'inline-block','verticalAlign':'top','width':'30%'}),

                 html.Div([
                 html.H3('Choose prediction period:'),
                 dcc.DatePickerRange(
                      id='date-picker-range',
                      min_date_allowed = datetime(2020, 2, 19),
                      max_date_allowed = datetime.today(),
                      start_date=datetime(2020, 2, 19),
                      end_date = datetime.today() - timedelta(days=1) + timedelta(days=30)
                )], style = {'display':'inline-block'}),

                html.Div([
                    html.Button(children='Predict',id='predict-button',n_clicks=0,style={'fontSize':24,'marginLeft':'30px'})
                ],style={'display':'inline-block'}),

                html.Div([
                html.H3('Choose a stock:',style = {'paddingRight':'30px'}),
                    dcc.Dropdown(
                        id='stocks-dropdown',
                        options=portfolio,
                        value=['AAPL'],
                        multi=False
                    )
                ], style = {'display':None,'verticalAlign':'top','width':'30%'}),


                html.Div([dcc.Graph(
                    id='my-graph',
                    figure = fig
                )],style={'width':'50%'}),

        html.Div(id='model_evaluation')

])


###########################################################################
###########################################################################

@app.callback(Output(component_id ='my-graph',component_property='figure'),
              [Input('predict-button','n_clicks')],
              [State('models-dropdown','value'),
               State('stocks-dropdown','value'),
               State('date-picker-range','start_date'),
               State('date-picker-range','end_date')])

def update_graph(n_clicks, value,value_stock ,start_date,end_date):
    start = datetime.strptime(start_date[:10], '%Y-%m-%d')
    end = datetime.strptime(end_date[:10], '%Y-%m-%d')

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    if value_stock == 'AAPL' and value == 'Random Forest Regression':
        # filename = 'random_forest.joblib'
        # loaded_rf_model = joblib.load(f'../models/' + filename)

        # index_split = int(len(data['adj close']) * 0.8), int(len(data['adj close']) * 0.2), int(len(data['adj close']) )

        split_date = start - timedelta(days=1)
        df_train = data.loc[:split_date]
        df_test = data.loc[start:]
        prediction_data = data.loc[start:end]
        df_train_x = df_train[['open', 'close', 'high', 'low']]
        df_train_y = df_train['adj close']

        num_split = len(df_train)
        num_test = len(df_test)
        num_prediction = len(prediction_data)

        model = RandomForestRegressor()
        rf_model = RandomForestRegressor(n_estimators=1000, random_state=2, min_samples_split=2, min_samples_leaf=1,
                                         max_depth=11, bootstrap=True)
        rf_model.fit(df_train_x, df_train_y)

        prediction = rf_model.predict(df_test[['open', 'close', 'high', 'low']])
        predictions = pd.DataFrame({"predictions": prediction}, index=df_test.index)

        fig = go.Figure(layout=layout)

        fig.add_trace(go.Scatter(y=data['adj close'][:num_split], x=data.index[:num_split], name='training data'))
        fig.add_trace(go.Scatter(y=data['adj close'][num_split:], x=data.index[num_split:], name='test data'))
        fig.add_trace(go.Scatter(y=predictions.predictions.loc[:end], x=data.index[:num_prediction], name='prediction RF'))

        fig.update_layout(title='APPLE Price- adjusted close ($)', xaxis_title="date",
                          yaxis_title="Stock price($)", )

    return fig
###########################################################################



@app.callback(Output('model_evaluation', 'children' ),
              [Input('predict-button','n_clicks')],
              [State('models-dropdown','value'),
               State('stocks-dropdown','value'),
               State('date-picker-range','start_date'),
               State('date-picker-range','end_date')])

def display_model_evaluation(n_clicks, value,value_stock ,start_date,end_date):
    if value_stock == 'AAPL' and value == 'Random Forest Regression':
        start = datetime.strptime(start_date[:10], '%Y-%m-%d')
        end = datetime.strptime(end_date[:10], '%Y-%m-%d')
        # filename = 'random_forest.joblib'
        # loaded_rf_model = joblib.load(f'../models/' + filename)
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        split_date = start - timedelta(days=1)
        df_train = data.loc[:split_date]
        df_test = data.loc[start:]
        df_train_x = df_train[['open', 'close', 'high', 'low']]
        df_train_y = df_train['adj close']

        prediction_length= len(data.loc[start:end])


        model = RandomForestRegressor()
        rf_model = RandomForestRegressor(n_estimators=1000, random_state=2, min_samples_split=2, min_samples_leaf=1,
                                         max_depth=11, bootstrap=True)
        rf_model.fit(df_train_x, df_train_y)

        prediction = rf_model.predict(df_test[['open', 'close', 'high', 'low']])
        predictions = pd.DataFrame({"predictions": prediction}, index=df_test.index)
        evaluation = calculate_results(predictions.predictions[:prediction_length], df_test['adj close'][:prediction_length])
        result_list = []
        for i in evaluation.items():
            result_list.append(f'{i[0]} : {i[1]}  , ')


    return result_list


###########################################################################
###########################################################################

if __name__ == "__main__":
    app.run_server(debug=True)

###########################################################################
###########################################################################
