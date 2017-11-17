# timeseries_prediction_LSTM
Time series prediction with LSTM

## Introduction

I learnt time series prediction in Operations Analytics class in SMU, we learnt ARIMA at that time and we used SAS EG to perform the analysis. SAS is not open source and is very expensive, not every company can buy SAS, but everyone can use python. After learning LSTM, a network suitable for sequence data. I'm always thinking about building a model to predict time series data. </br>
The differences between ARIMA and LSTM are as follow:
> 1. LSTM works better if we are dealing with huge amount of data and enough training data is available, while ARIMA is better for smaller datasets.
> 2. ARIMA requires a series of parameters (p,q,d) which must be calculated based on data, while LSTM does not require setting such parameters. However, there are some hyperparameters we need to tune for LSTM. 

## Methodology

The dataset I use is https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
1. Clean the data (drop null values, create dummy variable)
2. Change data to the format that is suitable for LSTM (reshape with window size， train test split)
3. Build and train the LSTM model
4. Evaluate the model on test set

## Result

#### time step = 1, epochs = 50: Test RMSE: 12.707 </br>
![timeseries](/data/loss_1.png) </br>
![timeseries](/data/predict_1.png) </br>

#### time step = 2, epochs = 50: Test RMSE: 39.480 </br>
![timeseries](/data/loss_2.png) </br>
![timeseries](/data/predict_2.png) </br>

#### time step = 3, epochs = 50: Test RMSE: 51.060 </br>
![timeseries](/data/loss_3.png) </br>
![timeseries](/data/predict_3.png) </br>

## Analysis

The result is unexpected. The expectation of increased performance with the increase of time steps was not observed. Dr. Jason Brownlee has a post analysing the influence of different time steps. The conclusion is that LSTM with time setp=1 peforms best, to train larger time steps, we may need to build larger network and train more epochs. See the reference for more information.

## References
https://machinelearningmastery.com/use-timesteps-lstm-networks-time-series-forecasting/ </br>
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ </br>
https://datascience.stackexchange.com/questions/12721/time-series-prediction-using-arima-vs-lstm </br>
