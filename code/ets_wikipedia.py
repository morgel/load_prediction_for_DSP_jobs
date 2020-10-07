import math
import copy
import time
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from numpy import array
import psutil

# for CPU profiling
profiler = psutil.Process()

# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(history, config, f_length):
    t,d,s,p,b,r = config
    # define model
    history = array(history)
    # ensure zero values are not present in actual observations, replace with very small value (dominik says this is reasonable)
    history[history == 0] = 1e-5
    model = ExponentialSmoothing(history, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p, initialization_method="estimated")
    # fit model
    model_fit = model.fit(optimized=True)
    #model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    # make one step forecast
    predictions = model_fit.predict(len(history), len(history) + f_length - 1)
    predictions = np.nan_to_num(predictions)
    return predictions

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return mean_squared_error(actual, predicted, squared=False)

# symmetric moving average percent error
def measure_smape(actual, forecast):
    y_true = actual.reshape(forecast.shape)
    top = np.abs(y_true - forecast)
    bottom = np.abs(y_true) + np.abs(forecast)
    bottom[bottom == 0] = 1
    return np.mean(top / bottom)
    
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]
    
# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # show all warnings and fail on exception if debugging
    if debug:
        # split dataset
        train, test = train_test_split(data, n_test)
        # fit model and make forecast for history
        predictions = exp_smoothing_forecast(train, cfg, n_test)
        smape_val = measure_smape(np.array(test), np.array(predictions))
        rmse_val = measure_rmse(test, predictions)
        result = (smape_val, rmse_val, predictions)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                # split dataset
                train, test = train_test_split(data, n_test)
                # fit model and make forecast for history
                predictions = exp_smoothing_forecast(train, cfg, n_test)
                smape_val = measure_smape(np.array(test), np.array(predictions))
                rmse_val = measure_rmse(test, predictions)
                result = (smape_val, rmse_val, predictions)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        #print(' > Model[%s] %.2f %.3f' % (cfg, result[0], result[1]))
        return (cfg, result[0], result[1], result[2])
    else:
        #return (key, None)
        return (cfg, None, None, None)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=False):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
    models = list()
    # define config lists
    t_params = ['add', 'mul', None]
    d_params = [True, False]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    b_params = [True, False]
    r_params = [True, False]
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t,d,s,p,b,r]
                            models.append(cfg)
    return models

df = pd.read_csv('Wikipedia_revisions_7D_1S.csv', sep='|', index_col=[0])

# configure for sampling rate
#df = df[df.index % 60 == 0]
#sample_rate = 60
#sample_rate_name = "1M"
df = df[15000:25080]
sample_rate = 1
sample_rate_name = "1S"

# create file with headers for results
results_file_path = f'ets_wikipedia_{sample_rate_name}_{time.time()}_results.csv'
f = open(results_file_path,'a+')
f.write(f"interval|f_length|p_number|s_number|mse|smape|gpu|util|duration\n")

vehicle_count = df['revisionCount'].tolist()

# global config values, window size, tracker for current offset and max prediction length
w_size = 3600
offset = 3600
f_max = 900

# training and test sets
y_train = vehicle_count[:offset]
y_test = vehicle_count[offset:offset + f_max]

train = pd.Series(y_train)
train.plot()
plt.show()

meta = []

# start recording metrics
start_time = time.time()
profiler.cpu_percent(interval=None)

# model configs
cfg_list = exp_smoothing_configs()
# grid search
scores = grid_search(vehicle_count[:offset+f_max], cfg_list, f_max)
#print('done')
# list top 3 configs
#for cfg, smape_val, rmse_val, predictions in scores[:3]:
    #print(cfg, smape_val, rmse_val, len(predictions))

util = (profiler.cpu_percent(interval=None) / psutil.cpu_count()) / 100 
end_time = time.time()
# calculate duration based on sample rate, always at least 1
duration = math.ceil((end_time - start_time) / sample_rate) 
#print(f"duration: {duration}")

cfg, smape_val, rmse_val, predictions = scores[0]

# update meta information: model, time of last model update, time when model updates is finished, number of observations
meta.append(cfg)
meta.append(offset)
meta.append(offset + duration)
meta.append(duration)

# update offset with duration 
offset += duration

interval_meta = dict.fromkeys([300, 900, 1800, 2700, 3600, math.inf])

# deep copy models for all interval lengths and update meta
for interval in interval_meta:
    
    temp = copy.deepcopy(meta)
    temp.append(offset + interval)
    temp.append(None)
    temp.append(None)
    temp.append(None)
    interval_meta[interval] = temp
    
    f.write(f"{interval}|{-1}|{offset}|{offset}|{-1}|{-1}|{0}|{util}|{end_time - start_time}|\n")
    #print(f"{interval}|{-1}|{offset}|{offset}|{-1}|{-1}|{0}|{util}|{end_time - start_time}|{interval_meta[interval]}\n")
    
# calculate number of observations based on sample rate
observations_count = math.floor((end_time - start_time) / sample_rate)

# calculate window of available observations based on offset
previous = offset
offset = offset + observations_count
#print(f"offset: {offset}")

# retrieve observations
#observations = vehicle_count[offset:offset + observations_count]
observations = vehicle_count[offset-w_size:offset]
#print(f"observations: {len(observations)} {offset-w_size} {offset}")

# evaluate forecast lengths by interval and write to file
for interval in interval_meta:    

    # update meta information, set next forecast and reset number of observations to 0
    interval_meta[interval][2] = interval_meta[interval][2] + duration
    interval_meta[interval][3] = 0
    
    for f_length in np.arange(90,990,90):

        smape_val = measure_smape(np.array(y_test[:f_length]), np.array(predictions[:f_length]))
        mse_val = measure_rmse(y_test[:f_length], predictions[:f_length])

        f.write(f"{interval}|{f_length}|{interval_meta[interval][1]}|{offset}|{mse_val}|{smape_val}|{0}|{util}|{end_time - start_time}\n")

interval_meta_copy = copy.deepcopy(interval_meta)

for i in range(offset, len(vehicle_count)):
               
    # test if enough datapoints exist for forecast
    if f_max > (len(vehicle_count) - i):
        break

    for interval in interval_meta_copy:

        # test if new model is available
        if (i == interval_meta_copy[interval][6]):
            interval_meta_copy[interval][0] = copy.deepcopy(interval_meta_copy[interval][5])
            interval_meta_copy[interval][5] = None
            interval_meta_copy[interval][1] = interval_meta_copy[interval][6]
            interval_meta_copy[interval][6] = None
            interval_meta_copy[interval][3] = interval_meta_copy[interval][7]
            interval_meta_copy[interval][7] = None

        # test if model needs to be updated
        if (i == interval_meta_copy[interval][4]):

            # retrieve training data for current window
            data = vehicle_count[i-w_size:i]

            # train new model
            start_time = time.time()
            profiler.cpu_percent(interval=None)
            
            # model configs
            cfg_list = exp_smoothing_configs()
            # grid search
            scores = grid_search(data, cfg_list, f_max)
            # select best performing model
            cfg, smape_val, rmse_val, predictions = scores[0]
            
            util = (profiler.cpu_percent(interval=None) / psutil.cpu_count()) / 100
            end_time = time.time()
            
            # calculate duration based on sample rate, always at least 1
            duration = math.ceil((end_time - start_time) / sample_rate) 

            # update meta information with next model info
            interval_meta_copy[interval][4] += interval  
            interval_meta_copy[interval][5] = copy.deepcopy(cfg)
            interval_meta_copy[interval][6] = interval_meta_copy[interval][2] + duration
            interval_meta_copy[interval][7] = duration

            f.write(f"{interval}|{-1}|{i}|{i}|{-1}|{-1}|{0}|{util}|{end_time - start_time}\n")
            #print(f"{interval}|{-1}|{i}|{i}|{-1}|{-1}|{0}|{util}|{end_time - start_time}\n")

        # test if forecast can be performed this interation
        if (i == interval_meta_copy[interval][2]):

            # perform update with observations and execute forecast
            observations = vehicle_count[i-interval_meta_copy[interval][3]:i]
            
            # retrieve training data for current window
            data = vehicle_count[i-w_size:i]
            
            # update the model with latest observations which arrived during 
            start_time = time.time()
            profiler.cpu_percent(interval=None)
            
            # make prediction based on time window
            try:
                # never show warnings too noisy
                with catch_warnings():
                    filterwarnings("ignore")
                    predictions = exp_smoothing_forecast(data, interval_meta_copy[interval][0], f_max)
            except:
                error = None
            
            util = (profiler.cpu_percent(interval=None) / psutil.cpu_count()) / 100
            end_time = time.time()
            
            # calculate duration based on sample rate, always at least 1
            duration = math.ceil((end_time - start_time) / sample_rate) 

            # update meta information, set next forecast
            interval_meta_copy[interval][2] = interval_meta_copy[interval][2] + duration

            actual = vehicle_count[i:i + f_max]

            # loop through forecast lengths and evaluate       
            for f_length in np.arange(90,990,90):
                
                smape_val = measure_smape(np.array(actual[:f_length]), np.array(predictions[:f_length]))
                mse_val = measure_rmse(actual[:f_length], predictions[:f_length])

                f.write(f"{interval}|{f_length}|{interval_meta_copy[interval][1]}|{i}|{mse_val}|{smape_val}|{0}|{util}|{end_time - start_time}\n")

            # reset number of observations
            interval_meta_copy[interval][3] = duration
        else:
            # increment observations while system is busy
            interval_meta_copy[interval][3] += 1

f.close()

