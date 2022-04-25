from datetime import datetime
from datetime import date,timedelta
import pandas as pd
import dash
from dash import dash_table
from dash import dcc
from dash import html
import plotly.graph_objs as go
import numpy as np
import joblib

from utilities.supportive_functions import preprocess, calculate_results, csv_to_dataset

from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing

import keras
from keras.models import Model
from keras import optimizers
from tensorflow.keras.optimizers import Adam
#
from keras.layers import Dense , Dropout , LSTM ,Activation , concatenate
from keras.layers import  Input as Input_keras

# ---------------------------------------

app = dash.Dash(__name__)


layout = go.Layout(
    yaxis=dict(domain=[1, 1]),
    legend=dict(traceorder='reversed'),
    yaxis2=dict(domain=[1, 1]),
    yaxis3=dict(domain=[1, 1])
)

fig = go.Figure(layout=layout)
###########################################################################

###########################################################################

all_options = {
    'LSTM': ['Deep Learning LSTM'],
#    'SARIMA': ['SARIMA'],
    'Random Forest Regression': ['Random Forest'],
    # 'XGBOOST Regression': ['XGBoost'],
    # 'Q-learning': ['Q-learning']
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

                html.Div(
                    [
                        dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
                        dbc.Progress(id="progress"),
                    ]
                ),

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
    fig = go.Figure(layout=layout)
    # reading data
    if value_stock == 'AAPL':
        df = preprocess(pd.read_csv(f'../data/ticker_data/AAPL_full_data.csv'))
    elif value_stock == 'MSFT':
        df = preprocess(pd.read_csv(f'../data/ticker_data/MSFT_full_data.csv'))
    elif value_stock == 'JNJ':
        df = preprocess(pd.read_csv(f'../data/ticker_data/JNJ_full_data.csv'))
    elif value_stock == 'CAT':
        df = preprocess(pd.read_csv(f'../data/ticker_data/CAT_full_data.csv'))
    elif value_stock == 'INTC':
        df = preprocess(pd.read_csv(f'../data/ticker_data/INTC_full_data.csv'))
    elif value_stock == 'VZ':
        df = preprocess(pd.read_csv(f'../data/ticker_data/VZ_full_data.csv'))
    elif value_stock == 'PFE':
        df = preprocess(pd.read_csv(f'../data/ticker_data/PFE_full_data.csv'))
    elif value_stock == 'TSLA':
        df = preprocess(pd.read_csv(f'../data/ticker_data/TSLA_full_data.csv'))



    start = datetime.strptime(start_date[:10], '%Y-%m-%d')
    end = datetime.strptime(end_date[:10], '%Y-%m-%d')

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    if  value == 'Random Forest Regression':

        df = df.iloc[:, :6]
        data = df.copy()
        data.date = pd.to_datetime(data.date)
        data = data.set_index('date')

        index_split = int(len(data['adj close']) * 0.8), int(len(data['adj close']) * 0.2), int(len(data['adj close']))
        df_train = data[:index_split[0]]
        df_test = data[index_split[0]:]
        df_train_x = df_train[['open', 'close', 'high', 'low']]
        df_train_y = df_train['adj close']


#####################################################
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

        if value_stock == 'AAPL':
            rf_model = RandomForestRegressor(n_estimators=1000, random_state=2, min_samples_split=2, min_samples_leaf=1,
                                           max_depth=11, bootstrap=True)
        elif value_stock == 'MSFT':
            rf_model = RandomForestRegressor(n_estimators=500, random_state=2, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=13, bootstrap=False)
        elif value_stock == 'JNJ':
            rf_model = RandomForestRegressor(n_estimators=500, random_state=1, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=10, bootstrap=False)
        elif value_stock == 'CAT':
            rf_model = RandomForestRegressor(n_estimators=50, random_state=3, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=13, bootstrap=False)
        elif value_stock == 'INTC':
            rf_model = RandomForestRegressor(n_estimators=20, random_state=12, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=13, bootstrap=False)
        elif value_stock == 'VZ':
            rf_model = RandomForestRegressor(n_estimators=3, random_state=15, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=16, bootstrap=False)
        elif value_stock == 'PFE':
            rf_model = RandomForestRegressor(n_estimators=50, random_state=13, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=10, bootstrap=False)
        elif value_stock == 'TSLA':
            rf_model = RandomForestRegressor(n_estimators=100, random_state=30, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=14, bootstrap=False)


        rf_model.fit(df_train_x, df_train_y)

        prediction = rf_model.predict(df_test[['open', 'close', 'high', 'low']])
        predictions = pd.DataFrame({"predictions": prediction}, index=df_test.index)


        fig.add_trace(go.Scatter(y=data['adj close'][:num_split], x=data.index[:num_split], name='training data'))
        fig.add_trace(go.Scatter(y=data['adj close'][num_split:], x=data.index[num_split:], name='test data'))
        fig.add_trace(go.Scatter(y=predictions.predictions.loc[:end], x=predictions.index[:num_prediction], name='prediction RF'))

        fig.update_layout(title='{} Price- adjusted close ($)'.format(value_stock), xaxis_title="date",
                          yaxis_title="Stock price($)", )

    ########################### LSTM #################

    ########### LSTM    ##########################################################################

    if value == 'LSTM':
        df_base = df.iloc[:, :6]
        df = df.iloc[:, 1:6]
        data = df.copy()

        df_base.date = pd.to_datetime(df_base.date)

        data_normaliser = preprocessing.MinMaxScaler()
        data_normalised = data_normaliser.fit_transform(data)

        history_points = 50

        # using the last {history_points} open high ,low, close, volume, adj close data points, predict the next adj close value

        ohlcv_histories_normalised = np.array(
            [data_normalised[i: i + history_points].copy() for i in range(len(data_normalised) - history_points)])
        next_day_adjclose_values_normalised = np.array([data_normalised[:, 4][i + history_points].copy() for i in
                                                        range(len(data_normalised) - history_points)])
        next_day_adjclose_values_normalised = np.expand_dims(next_day_adjclose_values_normalised, -1)

        next_day_adjclose_values = np.array(
            [np.array(data)[:, 4][i + history_points].copy() for i in range(len(data) - history_points)])
        next_day_adjclose_values = np.expand_dims(next_day_adjclose_values, -1)

        y_normaliser = preprocessing.MinMaxScaler()
        y_normaliser.fit(next_day_adjclose_values)

        ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
            f'../data/ticker_data/AAPL_full_data.csv')

        if value_stock == 'AAPL':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/AAPL_full_data.csv')
        elif value_stock == 'MSFT':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/MSFT_full_data.csv')
        elif value_stock == 'JNJ':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/JNJ_full_data.csv')
        elif value_stock == 'CAT':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/CAT_full_data.csv')
        elif value_stock == 'INTC':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/INTC_full_data.csv')
        elif value_stock == 'VZ':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/VZ_full_data.csv')
        elif value_stock == 'PFE':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/PFE_full_data.csv')
        elif value_stock == 'TSLA':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/TSLA_full_data.csv')

        ########################################
        index_split = int(len(data['adj close']) * 0.8), int(len(data['adj close']) * 0.2), int(len(data['adj close']))
        df_train = data[:index_split[0]]
        df_test = data[index_split[0]:]
        df_train_x = df_train[['open', 'close', 'high', 'low']]
        df_train_y = df_train['adj close']

        #####################################################
        split_date = start - timedelta(days=1)
        df_base = df_base.set_index('date')
        df_base.index = pd.to_datetime(df_base.index)
        df_train = df_base.loc[:split_date]
        df_test = df_base.loc[start:]
        prediction_data = df_base.loc[start:end]

        df_train_x = df_train[['open', 'close', 'high', 'low']]
        df_train_y = df_train['adj close']

        num_split = len(df_train)
        num_test = len(df_test)
        ######################################################

        df_train = ohlcv_histories[:num_split]
        df_test = ohlcv_histories[num_split:]

        y_train = next_day_adjclose_values[:num_split]
        y_test = next_day_adjclose_values[num_split:]

        unscaled_y_test = unscaled_y[num_split:]

        # model architecture

        lstm_input = Input_keras(shape=(history_points, 6), name='lstm_input')
        x = LSTM(50, name='lstm_0')(lstm_input)
        x = Dropout(0.2, name='lstm_dropout_0')(x)
        x = Dense(64, name='dense_0')(x)
        x = Activation('sigmoid', name='sigmoid_0')(x)
        x = Dense(1, name='dense_1')(x)
        output = Activation('linear', name='linear_output')(x)

        model = Model(inputs=lstm_input, outputs=output)
        adam = Adam(learning_rate=0.0005)
        model.compile(optimizer=adam, loss='mse')

        model.fit(x=df_train, y=y_train, batch_size=32, epochs=100, shuffle=True, validation_split=0.1)
        y_test_predicted = model.predict(df_test)
        y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

        # load, no need to initialize

        # filename = 'apple_ltsm_100epoch_basic.joblib'
        # loaded_lstm_model = joblib.load(f'../models/' + filename)
        #
        # y_test_predicted = loaded_lstm_model.predict(df_test)
        # y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

        print(list(df_base.index[num_split:]))
        print(list(unscaled_y_test.flatten()))
        print(list(y_test_predicted.flatten()))


        fig.add_trace(
            go.Scatter(y=data['adj close'][:num_split], x=list(df_base.index[:num_split]), name='training data'))
        fig.add_trace(
            go.Scatter(y=list(unscaled_y_test.flatten()), x=list(df_base.index[num_split:]), name='test data'))
        fig.add_trace(go.Scatter(y=list(y_test_predicted.flatten()), x=list(df_base.index[num_split:]),
                                 name='prediction test LSTM'))

        fig.update_layout(title='APPLE Price- adjusted close($),  LSTM model', xaxis_title="date",
        yaxis_title = "Stock price($)",


        )

    return fig
###########################################################################



@app.callback(Output('model_evaluation', 'children' ),
              [Input('predict-button','n_clicks')],
              [State('models-dropdown','value'),
               State('stocks-dropdown','value'),
               State('date-picker-range','start_date'),
               State('date-picker-range','end_date')])

def display_model_evaluation(n_clicks, value,value_stock ,start_date,end_date):
    result_list = []
    # reading data
    if value_stock == 'AAPL':
        df = preprocess(pd.read_csv(f'../data/ticker_data/AAPL_full_data.csv'))
    elif value_stock == 'MSFT':
        df = preprocess(pd.read_csv(f'../data/ticker_data/MSFT_full_data.csv'))
    elif value_stock == 'JNJ':
        df = preprocess(pd.read_csv(f'../data/ticker_data/JNJ_full_data.csv'))
    elif value_stock == 'CAT':
        df = preprocess(pd.read_csv(f'../data/ticker_data/CAT_full_data.csv'))
    elif value_stock == 'INTC':
        df = preprocess(pd.read_csv(f'../data/ticker_data/INTC_full_data.csv'))
    elif value_stock == 'VZ':
        df = preprocess(pd.read_csv(f'../data/ticker_data/VZ_full_data.csv'))
    elif value_stock == 'PFE':
        df = preprocess(pd.read_csv(f'../data/ticker_data/PFE_full_data.csv'))
    elif value_stock == 'TSLA':
        df = preprocess(pd.read_csv(f'../data/ticker_data/TSLA_full_data.csv'))

    if  value == 'Random Forest Regression':
        start = datetime.strptime(start_date[:10], '%Y-%m-%d')
        end = datetime.strptime(end_date[:10], '%Y-%m-%d')

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        split_date = start - timedelta(days=1)

        # reading data
        if value_stock == 'AAPL':
            df = preprocess(pd.read_csv(f'../data/ticker_data/AAPL_full_data.csv'))
        elif value_stock == 'MSFT':
            df = preprocess(pd.read_csv(f'../data/ticker_data/MSFT_full_data.csv'))
        elif value_stock == 'JNJ':
            df = preprocess(pd.read_csv(f'../data/ticker_data/JNJ_full_data.csv'))
        elif value_stock == 'CAT':
            df = preprocess(pd.read_csv(f'../data/ticker_data/CAT_full_data.csv'))
        elif value_stock == 'INTC':
            df = preprocess(pd.read_csv(f'../data/ticker_data/INTC_full_data.csv'))
        elif value_stock == 'VZ':
            df = preprocess(pd.read_csv(f'../data/ticker_data/VZ_full_data.csv'))
        elif value_stock == 'PFE':
            df = preprocess(pd.read_csv(f'../data/ticker_data/PFE_full_data.csv'))
        elif value_stock == 'TSLA':
            df = preprocess(pd.read_csv(f'../data/ticker_data/TSLA_full_data.csv'))

        df = df.iloc[:, :6]
        data = df.copy()
        data.date = pd.to_datetime(data.date)
        data = data.set_index('date')

        df_train = data.loc[:split_date]
        df_test = data.loc[start:]
        df_train_x = df_train[['open', 'close', 'high', 'low']]
        df_train_y = df_train['adj close']

        prediction_length= len(data.loc[start:end])

        model = RandomForestRegressor()

        if value_stock == 'AAPL':
            rf_model = RandomForestRegressor(n_estimators=1000, random_state=2, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=11, bootstrap=True)
        elif value_stock == 'MSFT':
            rf_model = RandomForestRegressor(n_estimators=500, random_state=2, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=13, bootstrap=False)
        elif value_stock == 'JNJ':
            rf_model = RandomForestRegressor(n_estimators=500, random_state=1, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=10, bootstrap=False)
        elif value_stock == 'CAT':
            rf_model = RandomForestRegressor(n_estimators=50, random_state=3, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=13, bootstrap=False)
        elif value_stock == 'INTC':
            rf_model = RandomForestRegressor(n_estimators=20, random_state=12, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=13, bootstrap=False)
        elif value_stock == 'VZ':
            rf_model = RandomForestRegressor(n_estimators=3, random_state=15, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=16, bootstrap=False)
        elif value_stock == 'PFE':
            rf_model = RandomForestRegressor(n_estimators=50, random_state=13, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=10, bootstrap=False)
        elif value_stock == 'TSLA':
            rf_model = RandomForestRegressor(n_estimators=100, random_state=30, min_samples_split=2, min_samples_leaf=1,
                                             max_depth=14, bootstrap=False)

        rf_model.fit(df_train_x, df_train_y)

        prediction = rf_model.predict(df_test[['open', 'close', 'high', 'low']])
        predictions = pd.DataFrame({"predictions": prediction}, index=df_test.index)
        evaluation = calculate_results(predictions.predictions[:prediction_length], df_test['adj close'][:prediction_length])

        for i in evaluation.items():
            result_list.append(f'{i[0]} : {i[1]}  , ')


    if value == 'LSTM':

        start = datetime.strptime(start_date[:10], '%Y-%m-%d')
        end = datetime.strptime(end_date[:10], '%Y-%m-%d')

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        split_date = start - timedelta(days=1)
        df_base = df.iloc[:, :6]
        df = df.iloc[:, 1:6]
        data = df.copy()

        df_base.date = pd.to_datetime(df_base.date)

        data_normaliser = preprocessing.MinMaxScaler()
        data_normalised = data_normaliser.fit_transform(data)

        history_points = 50

        # using the last {history_points} open high ,low, close, volume, adj close data points, predict the next adj close value

        ohlcv_histories_normalised = np.array(
            [data_normalised[i: i + history_points].copy() for i in range(len(data_normalised) - history_points)])
        next_day_adjclose_values_normalised = np.array([data_normalised[:, 4][i + history_points].copy() for i in
                                                        range(len(data_normalised) - history_points)])
        next_day_adjclose_values_normalised = np.expand_dims(next_day_adjclose_values_normalised, -1)

        next_day_adjclose_values = np.array(
            [np.array(data)[:, 4][i + history_points].copy() for i in range(len(data) - history_points)])
        next_day_adjclose_values = np.expand_dims(next_day_adjclose_values, -1)

        y_normaliser = preprocessing.MinMaxScaler()
        y_normaliser.fit(next_day_adjclose_values)

        ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
            f'../data/ticker_data/AAPL_full_data.csv')

        if value_stock == 'AAPL':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/AAPL_full_data.csv')
        elif value_stock == 'MSFT':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/MSFT_full_data.csv')
        elif value_stock == 'JNJ':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/JNJ_full_data.csv')
        elif value_stock == 'CAT':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/CAT_full_data.csv')
        elif value_stock == 'INTC':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/INTC_full_data.csv')
        elif value_stock == 'VZ':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/VZ_full_data.csv')
        elif value_stock == 'PFE':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/PFE_full_data.csv')
        elif value_stock == 'TSLA':
            ohlcv_histories, _, next_day_adjclose_values, unscaled_y, y_normaliser = csv_to_dataset(
                f'../data/ticker_data/TSLA_full_data.csv')

        ########################################
        index_split = int(len(data['adj close']) * 0.8), int(len(data['adj close']) * 0.2), int(len(data['adj close']))
        df_train = data[:index_split[0]]
        df_test = data[index_split[0]:]
        df_train_x = df_train[['open', 'close', 'high', 'low']]
        df_train_y = df_train['adj close']

        #####################################################
        split_date = start - timedelta(days=1)
        df_base = df_base.set_index('date')
        df_base.index = pd.to_datetime(df_base.index)
        df_train = df_base.loc[:split_date]
        df_test = df_base.loc[start:]
        prediction_data = df_base.loc[start:end]

        df_train_x = df_train[['open', 'close', 'high', 'low']]
        df_train_y = df_train['adj close']

        num_split = len(df_train)
        num_test = len(df_test)
        ######################################################

        df_train = ohlcv_histories[:num_split]
        df_test = ohlcv_histories[num_split:]

        y_train = next_day_adjclose_values[:num_split]
        y_test = next_day_adjclose_values[num_split:]

        unscaled_y_test = unscaled_y[num_split:]

        # model architecture

        lstm_input = Input_keras(shape=(history_points, 6), name='lstm_input')
        x = LSTM(50, name='lstm_0')(lstm_input)
        x = Dropout(0.2, name='lstm_dropout_0')(x)
        x = Dense(64, name='dense_0')(x)
        x = Activation('sigmoid', name='sigmoid_0')(x)
        x = Dense(1, name='dense_1')(x)
        output = Activation('linear', name='linear_output')(x)

        model = Model(inputs=lstm_input, outputs=output)
        adam = Adam(learning_rate=0.0005)
        model.compile(optimizer=adam, loss='mse')

        model.fit(x=df_train, y=y_train, batch_size=32, epochs=100, shuffle=True, validation_split=0.1)
        y_test_predicted = model.predict(df_test)
        y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)


        real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
        scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100

        evaluation = {}
        evaluation['real_mse'] = real_mse
        evaluation['scaled_mse'] = scaled_mse

        for i in  evaluation.items():
            result_list.append(f'{i[0]} : {i[1]}  , ')
    return result_list


###########################################################################
@app.callback(
    [Output("progress", "value"), Output("progress", "children")],
    [Input("progress-interval", "n_intervals")],
)
def update_progress(n):
    # check progress of some background process, in this example we'll just
    # use n_intervals constrained to be in 0-100
    progress = min(n % 110, 500)
    # only add text after 5% progress to ensure text isn't squashed too much
    return progress, f"Model is being trained, wait a bit..{progress} %" if progress >= 5 else ""
###########################################################################

if __name__ == "__main__":
    app.run_server(debug=True)

###########################################################################
###########################################################################
