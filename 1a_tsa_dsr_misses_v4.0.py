#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:02:28 2020

@author: divyeg

Business Problem: Calculate Goals for improvement of DSR metric at segment level for targetted improvements.

Data Science Formulation: Forecast DSR misses (dsr numerator) for the next four weeks in the future using historical data.

Why Forecast DSR Misses?: DSR (Delivery Success Rate) is a key SDS metric used to drive performance discussions among senior leadership. Both program 
and operations team try to improve this metric using variety of initiatives and experiment lanches aiming to improve the metric. Goals 
are calculated by senior leadership in a non scientific way to challenge the program and operations teams for finding improvement opportunities
In order to keep Customer Service standards high, the stakeholdders should calculate the goals based on the time series trends of the metric. If the 
metric is already up 400 bps at the end of year, the YTD numbers would be averaged for the whole year and without any initiative, the metric
will increase based on the historical trends because Denominator of the metric is increasing with time. Hence, Goals should be calculated based on
forecasted values of DSR numerator and denominator. DSR denominator is forecasted by CP team. In this code, we try to forecast the numerator.

Methodology: Forecasting future DSR misses will prepare team managers for risky situations that have already occured in the past and will help
senior leadership to successfully articulate goals based on scientific model vs average based analytical model. The idea behind forecasting 
this number accutately is to capture the trend in the metric. To begin with, data was collected for DSR numerator at daily level and exploratory
analysis is done to find visual evidence that DSR numerator is a function of time t. 
After establishing that DSR(num) follows the equation: ∆y=α+βt+γy_(t-1)+ δ_1 ∆y_(t-1))+⋯+ δ_(p-1) ∆y_(t-(p-1) )+ ∈_t, time series decomposition
was performed to separate trends, seasonality and residual. XGBoost and SARIMAX model was tried to fit and predict the DSR misses.

"""

# Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(4)

# A. Getting and preparing the data
# 1. Importing and manipulating the data
#%%============================================================================== 
data = pd.read_csv("/Users/divyeg/Downloads/1 Delivered:Ongoing_Work/DSR TSA/dsr_metric_daily.csv", index_col=0)
data['disconnect_day'] = data['disconnect_day'].apply(lambda x: x.split(' ')[0])
data.index = pd.to_datetime(data.disconnect_day.values)
data = data.sort_index()
data.index.freq = pd.tseries.frequencies.to_offset("D")
data['year'] = data['disconnect_day'].apply(lambda x: x.split('-')[0])
data['month'] = data['disconnect_day'].apply(lambda x: x.split('-')[1])
data['day'] = data['disconnect_day'].apply(lambda x: x.split('-')[2])
data['weekday'] = pd.to_datetime(data['disconnect_day']).apply(lambda x: (x.weekday()))
data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x==0 or x==1 else 0)

pvars = ['dsr_denom', 'dsr_num', 'dsr'] #naming the three series
for var in pvars:
    globals()['series_' + str(var)] = data[var]
#%%============================================================================== 
#================================================================================
# B. Exploratory Data Analysis
#%%==============================================================================
# 1. Plotting ACF and PACF plot to test stationarity
from statsmodels.graphics import tsaplots
fig, axes = plt.subplots(2,1, figsize=(12.8, 9.2), dpi=100)
tsaplots.plot_acf(series_dsr, ax=axes[0],lags = 49, alpha = 0.05, title="Autocorrelation plot for 49 lags")
tsaplots.plot_pacf(series_dsr, ax = axes[1],lags = 49, alpha = 0.05, method = 'ols', title="Partial Autocorrelation plot for 49 lags")
fig.savefig("/Users/divyeg/Downloads/1 Delivered:Ongoing_Work/DSR TSA/image files/ACF and PACF.jpg", bbox_inches='tight')
plt.close()

#kpss test for stationarity
from statsmodels.tsa.stattools import kpss
print("KPSS Test p-value",kpss(series_dsr, regression='c', nlags=15)[1]) #since p-value > 0.05, series is stationary and differencing is not required
from statsmodels.tsa.stattools import adfuller
print("ADFF Test p-value",adfuller(series_dsr, regression='c', maxlag=15)[1]) #since p value < 0.05 series is stationary and differecning is not required

#applying differencing to the series
from statsmodels.tsa.statespace import tools
series_diff = tools.diff(series_dsr, k_diff=1) #kpss p value > 0.05 for k_diff = 1
fig, axes = plt.subplots(2,1, figsize=(12.8, 9.2), dpi=100)
tsaplots.plot_acf(series_diff, ax=axes[0],lags = 49, alpha = 0.05, title="Autocorrelation plot for 49 lags with diff=1")
tsaplots.plot_pacf(series_diff, ax = axes[1],lags = 49, alpha = 0.05, method = 'ols', title="Partial Autocorrelation plot for 49 lags with diff=1")
fig.savefig("/Users/divyeg/Downloads/1 Delivered:Ongoing_Work/DSR TSA/image files/ACF and PACF diff=1.jpg", bbox_inches=  'tight')
plt.close()

print("KPSS Test p-value", kpss(series_diff, regression='c', nlags=15)[1])
print("ADFF Test p-value", adfuller(series_diff, regression='c', maxlag=15)[1])

#plotting original vs differenced series
plt.figure(2, figsize=[12.8,9.2], dpi=100)
plt.subplot(211)
plt.title("DSR Series WW")
plt.plot(range(len(series_dsr)), series_dsr, '-k')
plt.subplot(212)
plt.title("DSR Serices diff =2 WW")
plt.plot(range(len(series_diff)), series_diff, '-k')
plt.savefig("/Users/divyeg/Downloads/1 Delivered:Ongoing_Work/DSR TSA/image files/series vs diff_series.jpg", bbox_inches=  'tight')
plt.close()

#%%============================================================================
# 2. Decomposing the time series
#==============================================================================
from statsmodels.tsa.seasonal import STL
for var in pvars:
    globals()['result_'+ str(var)] = STL(data[var],
                                        period =7,
                                        seasonal = 91,
                                        trend = 89,
                                        seasonal_deg = 0,
                                        trend_deg = 1,
                                        robust = True).fit()
    
    globals()['trend_strength_' + str(var)] = max(0.0,1 - (globals()['result_'+ str(var)].resid.var() / (data[var] - globals()['result_'+ str(var)].seasonal).var())) #strength of trend
    globals()['seasonal_stength_' + str(var)] = max(0, 1 - (globals()['result_'+ str(var)].resid.var() / (data[var] - globals()['result_'+ str(var)].trend).var())) #strength of seasonality
    data['expected_value_'+str(var)] = globals()['result_'+ str(var)].trend + globals()['result_'+ str(var)].seasonal
    
# 3. Plotting the decomposed time series for DSR
#==============================================================================
series = series_dsr
result = result_dsr
plt.figure(5, figsize=[19.2,13.8], dpi=100)
plt.subplot(411)
plt.title("Decomposition of Observed Series")
plt.plot(range(len(series)), series, '-k')
plt.plot(range(len(series)), series-result.resid)
plt.subplot(412)
plt.title("Trend of the series")
plt.plot(range(len(result.trend)), result.trend, '-k')
plt.subplot(413)
plt.title("Seasonality of the series")
plt.plot(range(len(result.seasonal)), result.seasonal, '-k')
plt.subplot(414)
plt.title("Residuals of the series")
plt.plot(range(len(result.resid)), result.resid, '-k')
plt.savefig('/Users/divyeg/Downloads/1 Delivered:Ongoing_Work/DSR TSA/image files/series_dsr_decomposed.jpg', bbox_inches='tight')
plt.close()

#%%============================================================================
del var, series, pvars, result

# 2. Detecting outliers in residuals using laney-p control charts
#================================================================================
def z_score(series):
    p_bar = (np.sum(series))/len(series)
    sigma_bar = np.std(series)
    #z_i = np.array([(series1[i] - p_bar) / sigma_bar for i in range(len(series1))])
    ucl = np.repeat(p_bar + 2*sigma_bar, len(series))
    lcl = np.repeat(p_bar - 2*sigma_bar, len(series))
    return ucl, lcl, np.repeat(p_bar, len(series))

data_diff = data[1:]
data_diff['dsr_diff'] = series_diff
data_diff['ucl'], data_diff['lcl'], data_diff['p_bar'] = z_score(series_diff)
data_diff['is_outlier'] = np.where((data_diff['lcl']>data_diff['dsr_diff']) | (data_diff['ucl']<data_diff['dsr_diff']), 1,0)
#%%==============================================================================
# 3. Creating exog variables
data_diff['weekday'] = data_diff['weekday'].apply(lambda x: str(x))
X = data_diff[['year', 'day', 'weekday']]
X = pd.get_dummies(X, drop_first = True)
X = pd.concat([X, data_diff[['is_weekend']]], axis=1)
#Forecasting the differenced time series

#%% ===========================================================================
Y = series_diff
X_copy = X
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_copy, Y, test_size = 0.1, random_state = 101, shuffle = False)
#%%============================================================================
#1. Model - XGBOOST============================================================
import xgboost as xgb
model_xgb = xgb.XGBRegressor(alpha=10, base_score=0.5, booster='dart', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.3, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=5, min_child_weight=1, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0.25, reg_lambda=0.95, scale_pos_weight=1,
             seed=None, silent=None, subsample=1, verbosity=1)
model_xgb.fit(X_train, Y_train)
train_pred_xgb = pd.Series(model_xgb.predict(data = X_train), index = Y_train.index, name='train_pred')
test_pred_xgb = pd.Series(model_xgb.predict(data = X_test), index = Y_test.index, name = 'test_pred')

from sklearn import metrics
print("train RMSE XGB = ", np.round(np.sqrt(metrics.mean_squared_error(Y_train, train_pred_xgb)),4))
print("test RMSE XGB = ", np.round(np.sqrt(metrics.mean_squared_error(Y_test, test_pred_xgb)),4))
print()
print("train MAPE XGB = ", np.round(np.sqrt(metrics.mean_absolute_error(Y_train, train_pred_xgb))*100,4),'%')
print("test MAPE XGB = ", np.round(np.sqrt(metrics.mean_absolute_error(Y_test, test_pred_xgb))*100,4), '%')

resid_data = pd.concat([Y_train, test_pred_xgb], axis = 0)

plt.figure(1, figsize = [12.8,9.2], dpi = 100)
plt.subplot(111)
plt.title("forecasting using XGBOOST")
plt.plot(range(len(Y_test)), Y_test, '-k')
plt.plot(range(len(test_pred_xgb)), test_pred_xgb, '--b')
plt.savefig('/Users/divyeg/Downloads/1 Delivered:Ongoing_Work/DSR TSA/image files/DSR Forecasted XGB.jpg', bbox_inches='tight')
plt.close()
#%% ===========================================================================
#3. Model - SARIMAX
# =============================================================================
from statsmodels.tsa.statespace import sarimax
model = sarimax.SARIMAX(Y_train, exog=X_train, order=(4,1,7), enforce_stationarity=False, enforce_invertibility=False)
result = model.fit(low_memory=True, maxiter=100, disp=0)
print(result.aic)
train_pred_sarimax = result.get_forecast(steps = 729, exog = X_train).summary_frame(alpha = 0.1)
test_pred_sarimax = result.get_forecast(steps = 82, exog = X_test).summary_frame(alpha = 0.1)
print("train MAPE SARIMAX = ", np.round(np.sqrt(metrics.mean_absolute_error(Y_train, train_pred_sarimax['mean']))*100,4),'%')
print("test MAPE SARIMAX = ", np.round(np.sqrt(metrics.mean_absolute_error(Y_test, test_pred_sarimax['mean']))*100,4), '%')

plt.figure(1, figsize=[12.8,9.2], dpi = 100)
plt.title("Performance of SARIMAX Model on Test Set")
plt.plot(range(len(Y_test)), Y_test, '-k')
plt.plot(range(len(test_pred_sarimax)), test_pred_sarimax['mean'], '--b')
plt.plot(range(len(test_pred_sarimax)), test_pred_sarimax['mean_ci_lower'], '--r')
plt.plot(range(len(test_pred_sarimax)), test_pred_sarimax['mean_ci_upper'], '--r')
plt.savefig('/Users/divyeg/Downloads/1 Delivered:Ongoing_Work/DSR TSA/image files/DSR Forecasted SARIMAX.jpg', bbox_inches="tight")
plt.close()
#%% ===========================================================================
#reversing the differencing
temp = test_pred_sarimax['mean']
actual = series_dsr[729:811]
temp = temp+actual
#temp = temp.apply(lambda x: x + actual)
temp = temp.dropna()
print("test MAPE SARIMAX Actual= ", np.round(np.sqrt(metrics.mean_absolute_error(series_dsr[730:811], temp))*100,4), '%')

plt.figure(1, figsize=[12.8,9.2], dpi = 100)
plt.title("Performance of SARIMAX Model on Test Set")
plt.plot(range(len(temp)), series_dsr[730:811], '-k')
plt.plot(range(len(temp)), temp, '--b')
plt.savefig('/Users/divyeg/Downloads/1 Delivered:Ongoing_Work/DSR TSA/image files/DSR Forecasted SARIMAX actual2.jpg', bbox_inches="tight")
plt.close()

del temp, actual

data_diff.to_excel("/Users/divyeg/Downloads/1 Delivered:Ongoing_Work/DSR TSA/model_data.xlsx")
#%%==============================================================================



#archived code===================================================================
# #FFT for extrapolating trend
# from numpy import fft
# def fourierExtrapolation(x, n_predict):
#     n = x.size
#     n_harm = 10                     # number of harmonics in model
#     t = np.arange(0, n)
#     #p = np.polyfit(t, x, 1)         # find linear trend in x
#     x_notrend = x #- p[0] * t        # detrended x
#     x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
#     f = fft.fftfreq(n)              # frequencies
#     indexes = range(n)
#     # sort indexes by frequency, lower -> higher
#     #indexes.sort(key = lambda i: np.absolute(f[i]))
#  
#     t = np.arange(0, n + n_predict)
#     restored_sig = np.zeros(t.size)
#     for i in indexes[:1 + n_harm * 2]:
#         ampli = np.absolute(x_freqdom[i]) / n   # amplitude
#         phase = np.angle(x_freqdom[i])          # phase
#         restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
#     return restored_sig + p[0] * t
# 
# x = result_dsr.trend[0:730]
# n_predict = 100
# f = fft.fft(x)
# extrapolation = fourierExtrapolation(X_train, n_predict)
# plt.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label = 'extrapolation')
# plt.plot(np.arange(0, x.size), x, 'b', label = 'x', linewidth = 3)
# plt.legend()
# plt.show()
# plt.close()
# =============================================================================
#Archived Code
# =============================================================================
# series1 = data.dsr_num[data.year == '2018']
# series2 = data.dsr_num[data.year == '2019']
# series3 = data.dsr_num[data.year == '2020']
# series4 = data.dsr_num
# =============================================================================

# =============================================================================
# transforming the series for decomposition
# def invboxcox(y,ld):
#    if ld == 0:
#       return(np.exp(y))
#    else:
#       return(np.exp(np.log(ld*y+1)/ld))
# =============================================================================

# =============================================================================
# for var in pvars:
#     series_temp, lamda, alpha = stats.boxcox(globals()['series_' + str(var)], alpha = 0.05) 
#     if lamda < 1:
#         globals()['series_'+str(var)] = np.log(globals()['series_' + str(var)])
#         globals()['transform_'+str(var)] = 'log'
#     else:
#         globals()['transform_'+str(var)] = 'no_transform'
#     del series_temp, alpha, lamda
# =============================================================================
# =============================================================================
# If 0 < lamda < 1, model is multiplicative else additive. 
# If model is multiplicative, use box-cox transformation to convert it into an additive model
# =============================================================================
# data_weekly = data.groupby('week').agg({'dsr_num':'sum', 'dsr_denom':'sum', 'expected_value_dsr_num':'sum', 'expected_value_dsr_denom':'sum'})
# data_weekly['dsr'] = data_weekly.dsr_num / data_weekly.dsr_denom 
# data_weekly['dsr_expected'] = data_weekly.expected_value_dsr_num / data_weekly.expected_value_dsr_denom
# 
# plt.figure(6, figsize=[19.2, 13.8], dpi=100)
# plt.subplot(111)
# plt.title("Weekly actual dsr vs expected value of dsr")
# plt.plot(range(len(data_weekly.dsr)), data_weekly.dsr, '-k')
# plt.plot(range(len(data_weekly.dsr_expected)), data_weekly.dsr_expected)
# plt.savefig("Weekly_actual_expected_dsr.jpg", bbox_inches='tight')
# plt.close()
# =============================================================================
#%%

