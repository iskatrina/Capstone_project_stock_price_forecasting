# Capstone project: Stock price forecasting


## Overview

This project is part of Udacity Data Science project + some additions which I used or plan to use in the project to accompany it for further development . 
Project offers few models which could be used to predict Stock prices for number of Stocks choosen for Portfolio within choosen timeframe. Each model shows its accuracy in each case.
Using Stock price forecast , could help develop a strategy for Buying/Selling  Stocks for possible profit. 


## Instructions

- Run the Dashapp_1.py
- choose Stock name, model which you would like to use,  date range for prediction (1 day, 2,3,..30 days)
- Press Predict button.

PS. in case of LSTM Deep learning model, you need to wait to allow model retrain data according to choosen time period


Go to http://127.0.0.1:8050/

## Problem Introduction

Prediction of stock price movement is regarded as a challenging task of financial time series prediction. An accurate prediction of stock price movement may yield profits for investors. Due to the complexity of stock market data, development of efficient models for predicting is very difficult. 
The traditional models follow a stochastic probabilistic approach, while more recent models are based on machine learning methods. 
My ultimate goal is to create prediction which would help gain profit by trading stock by usig both machine learning methods and stochastic approach.


## Strategy to solve the problem

Basic strategy is to buy stock shares when the price is low, and sell them later when the price is higher. 
Goal: create predictive model/models of  price movements of a stock as accurate as possible and help make more data informed decision while buying or selling stock to gain a profit. 

The tasks involved are the folowing:

* Retrieve Data from financial sites:
        <br>*  obtain information about stocks from S&P 500.
	<br>*   retrive fundanemtal data of S&P 500 stocks from Qandle.
        <br>*  Aquire pricing info for list of stocks from yFinance
	<br>* Combining data together
* Feature engeneering
* Choosing Stocks from S&P 500 to build Portfolio
* Optimize Porfolio. Define Optimal Risky Portfolio
* ML model creation:
        <br> * using DeepLearning LSTM
        <br> *  using Random Forest
        <br> *  using SARIMA
* Build interactive application using Dash


## Metrics

The error metrics were calculated over the test dataset for predicted next-day adjusted close. These predictions were scored on  
 - MSE (Mean Squared Error)  to indicate magnitude of error.

<img width="519" alt="formula MSE - Google Search 2022-05-07 15-33-38" src="https://user-images.githubusercontent.com/15786410/167260522-3d9f2724-7641-4fd6-a8ee-6ceab75f6ab1.png">


## EDA


## Forecastig Models

Ive created several models to compare performance and accuracy of each.

        <br> *  Random Forest
        <br> *  SARIMA
        <br> *  DeepLearning LSTM


## Hyperparameter tuning

To define best performing parameters for each model I used hyperparameter tuning:

 for Random Forest model was used RandomizedSearchCV and search was performed on the following parameters:
* n_estimators — number of trees in the forest
* max_depth — maximum depth in a tree
* min_samples_split — minimum number of data points before the sample is split
* min_samples_leaf — minimum number of leaf nodes that are required to be sampled
* bootstrap — sampling for data points, true or false
* random_state — generated random numbers for the random forest.


For ARIMA/SARIMA model to hypertune parameters was used pmdarima.auto_arima() 


##  Results

A Dash application where user can choose a stock from constructed Portfolio and choose the Model with which user can predict stock price. As well as accuracy (MSE)


<img width="964" alt="Dash 2022-04-25 22-35-32" src="https://user-images.githubusercontent.com/15786410/165170593-fabdad7b-6db5-4a2a-8e1a-ea54d21121f2.png">




## Conclusion/Reflection

Forecasting models show high potential and could be developed and improved further. 



## Improvements

Prediction models perform descent but far from been perfect. And there is number of improvements could be made:
- construct and use in the model additional features , ultimatly improving accuracy of prediction
- develop algorythm to define and mark points in time when it is recommended to buy stock and when to sell. Depict it in a chart
- backtest algorythm and model performance

- 


