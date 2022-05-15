# Capstone project: Stock price forecasting

## Project Definition

### Project Overview

This project is part of Udacity Data Science project and envolves creating ML model and predict future market fluctuations of a Stock. 
It envolves getting the market data, analysing them and creating forecaseting of Stock Price, in particular Adj Close Price, which is the goal of a project.
Project consists of several parts. First is obtaining the data of market historical movements for stocks from S&P 500 and defining Portfolio of Stocks using Clustering techniques along with Domain techniques. Then Project explore and visualize data to understand them better. As a result, project provides few ML models which predict Stock price any Stock of Portfolio. Each model evaluaton based on its accuracy.
Using Stock price forecast  could help to develop the progect further and buid a strategy for Buying/Selling Stocks for possible profit. 

Prediction of stock price movement is regarded as a challenging task of financial time series prediction. An accurate prediction of stock price movement may yield profits for investors. Due to the complexity of stock market data, development of efficient models for predicting is very difficult. 
The traditional models follow a stochastic probabilistic approach, while more recent models are based on machine learning methods. 
My ultimate goal is to create prediction which would help gain profit by trading stock by using  machine learning methods .
In this work I aim to predict Adjusted Close Price of a stock with focus on first 1,2,7,14 up to 30 days and gain insight about its accuracy with help of two ML models - Random Forest and DeepLearning LSTM.

The approach, the methodology and the results are documented here. The
conclusion is that Random Forest method is the most accurate algorithm to Forecast stock price but DeepLearning LSTM shows very high potential. 

The project description is available from Udacity at this link - 



### Problem Statement
 Project doesnt provide any market data and I have to obtain them myself. Based on data exploration I need to choose best suitable forecasting technique and predict Stock Price and in particulary Adj Close. Evaluation of prediction based on accurasy metric, I choosed MSE metric for my ML models.

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
 
* Build interactive application using Dash


### Metrics

Two forecast models were created and trained on a train dataset. Afterwards, each model performance was tested on a Test Dataset and its accuracy has been measured. 
The error metric ( MSE (Mean Squared Error) ) was choosen as a main metric for measuring the accuracy of prediction, calculated over the test dataset for predicted next-day Adjusted Close, to indicate magnitude of error. It provides a quadratic loss function and that it also measures of the uncertainty in forecasting.

<img width="519" alt="formula MSE - Google Search 2022-05-07 15-33-38" src="https://user-images.githubusercontent.com/15786410/167260522-3d9f2724-7641-4fd6-a8ee-6ceab75f6ab1.png">

## Analysis

### Data Exploration and Data Visualization
 Stock tickers from S&P 500 were obtained, and extended by retriving fundanemtal data of S&P 500 stocks from Qandle.
 Aquired pricing info for list of stocks from yFinance has been merged with thosedata together.
 
 Exloration of quality of data showed that Data quality is just perfect and ready for further use by ML algorthms. 
 EDA also gave an idea/visualization about how  Adj Close Price and Volume of each stock is distributed.
 
 


## Methodology

### Data Preprocessing

No further data preprocessing required since all data obtained with pipeline showed high quality .

### Implementatoin/ Forecastig Models

Initial setup

PyCharm and Jupyter Notebook were used to code the project in Python.



This py file obtains data of historical market Prices: https://github.com/iskatrina/Capstone_project_stock_price_forecasting/blob/main/Capstoneproject_1_data_retrieval.py
 - function sp500_list_retrieval() helps to scrap ticker names of S&P 500 from Wikipedia 
 - function fundamental_data_pull() helps to obtain fundamental data from Quandle
 - pricing_yfinance_data() helps to pull data from yfinance 
 - and pipeline function - function data_collection() -  performs all pipeline and combines data together , which eventually saved into dataframes. Each dataframe contains information about each stock.


For further exploration, portfolio construction and predictions were used number of functions-helpers, which you can find in py file : https://github.com/iskatrina/Capstone_project_stock_price_forecasting/blob/main/utilities/supportive_functions.py

- function preprocess() - helps to prepare and extract needed data for further Analysis and predictions.
- function calculate_results() - help to evaluate the accuracy of Predictiv model
- function  csv_to_dataset() - helps to prepare and normilize data and separate in needed parts for Predictive model using DeepLearning LSTM technique.


Using Modul ta , Ive engeneered additional techinical features for each stock.  Although I didnt use them in my Forecasting models yet, it gave me good foundation  for further extension of a project. Yuo can find it here: https://github.com/iskatrina/Capstone_project_stock_price_forecasting/blob/main/Notebooks/2_Capstoneproject_feature_engeneering.ipynb

ALthough Udacity project didnt had it in requirenments, Ive implemented Cluster algorythm to constract  Portfolio from S&P 500 stocks, and you can find it here: https://github.com/iskatrina/Capstone_project_stock_price_forecasting/blob/main/Notebooks/4_Capstoneproject_Stock_choice_to_build_Portfolio.ipynb
Also I optimized Portfolio using Efficient Frontier and Sharpe Ratio techniques, you can find Notebook here: 
https://github.com/iskatrina/Capstone_project_stock_price_forecasting/blob/main/Notebooks/5_CapstoneProject_PortfolioOptimization_TargetPortfolio.ipynb 


For the main part of this project, Ive created several models to compare performance and accuracy of each. I included into submittion only two:

        <br> *  Random Forest ()
	https://github.com/iskatrina/Capstone_project_stock_price_forecasting/blob/main/Notebooks/8_1_CapstoneProject_ML_RandomForest.ipynb
        <br> *  DeepLearning LSTM https://github.com/iskatrina/Capstone_project_stock_price_forecasting/blob/main/Notebooks/6_4_CapstoneProject_ML_modeling_DeepLearning_LSTM.ipynb


### Refinement
### Hyperparameter tuning

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

### Model Evaluation and Validation

A Dash application where user can choose a stock from constructed Portfolio and choose the Model with which user can predict stock price. As well as accuracy (MSE)


<img width="964" alt="Dash 2022-04-25 22-35-32" src="https://user-images.githubusercontent.com/15786410/165170593-fabdad7b-6db5-4a2a-8e1a-ea54d21121f2.png">

- The results for Random Forest on the test set are actually the best after hyperparameters tuning, if not tied for, on every stock. This is true for MSE. 
- LSTM model shows descent results and shows high potential , although performing less accurate then Random Forest.

### Justification

## Conclusion
### Reflection

	Both forecasting models show high potentialin predicting Adj Close stock price within first 30 days of prediction and could be developed and improved further to serve the purpose of decision making - to buy or to sell particular stock. Since most important is the  accuracy of stock movement itself(up or down) and less the numbers in this simple strategy, these models could be enough to make this desicion and both models shows that it is realistic.
	
### Improvement	


## Deliverables

- Application
- GitHub Repository


## Instructions

- Run the Dashapp_1.py
- choose Stock name, model which you would like to use,  date range for prediction (1 day, 2,3,..30 days)
- Press Predict button.

PS. in case of LSTM Deep learning model, you need to wait to allow model retrain data according to choosen time period
	
  For further deveopment and improvements of models , more features should be designed and included in model prediction, and  technical indicators also could help accompany those desicion during buying or selling stock.
  In this work accuracy of a model was checked only on the test set data and was not proved on the data exciding available dataset , which could bring extra challenges to the process.



## Improvements

Prediction models perform descent but far from been perfect. And there is number of improvements could be made:
- construct and use in the model additional features , ultimatly improving accuracy of prediction
- develop algorythm to define and mark points in time when it is recommended to buy stock and when to sell. Depict it in a chart
- backtest algorythm and model performance




