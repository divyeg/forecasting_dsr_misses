# Forecasting DSR Misses

## Business Problem: 
Calculate Goals for improvement of DSR metric at segment level for targetted improvements.

## Data Science Formulation: 
Forecast DSR misses (dsr numerator) for the next four weeks in the future using historical data.

### Why Forecast DSR Misses?: 
DSR (Delivery Success Rate) is a key SDS metric used to drive performance discussions among senior leadership. Both program 
and operations team try to improve this metric using variety of initiatives and experiment lanches aiming to improve the metric. Goals 
are calculated by senior leadership in a non scientific way to challenge the program and operations teams for finding improvement opportunities
In order to keep Customer Service standards high, the stakeholdders should calculate the goals based on the time series trends of the metric. If the 
metric is already up 400 bps at the end of year, the YTD numbers would be averaged for the whole year and without any initiative, the metric
will increase based on the historical trends because Denominator of the metric is increasing with time. Hence, Goals should be calculated based on
forecasted values of DSR numerator and denominator. DSR denominator is forecasted by CP team. In this code, we try to forecast the numerator.

### Methodology: 
Forecasting future DSR misses will prepare team managers for risky situations that have already occured in the past and will help
senior leadership to successfully articulate goals based on scientific model vs average based analytical model. The idea behind forecasting 
this number accutately is to capture the trend in the metric. To begin with, data was collected for DSR numerator at daily level and exploratory
analysis is done to find visual evidence that DSR numerator is a function of time t. 
After establishing that DSR(num) follows the equation: ∆y=α+βt+γy_(t-1)+ δ_1 ∆y_(t-1))+⋯+ δ_(p-1) ∆y_(t-(p-1) )+ ∈_t, time series decomposition
was performed to separate trends, seasonality and residual. XGBoost and SARIMAX model was tried to fit and predict the DSR misses.

### Result: 
The project was shutdown because my team thinks it is politically incorrect to calculate accurate forecasts and that if I leave the team
who will manage these solutions. This project was done out of passion and a lot of self learning was achieved.
