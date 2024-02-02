# Imports
import ann
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras

import cv2

import math
import numpy as np
import pandas as pd
from scipy import interpolate
import statistics as stt

from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns

from datetime import datetime
import pickle
import re

import customization
import data_fetching
import hyperparameter_optimization as HPO
import utilities
import data_paths

# Removes row and column display limit for easier debugging
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def interrupt(message='Interrupted by interrupt() function.'):
    assert False, message


def heatmap_compare(models: dict) -> None:
    model_names = ''
    dates = utilities.dt_list

    index = 0
    for _, model in models.items():

        preds = model["obj"].predict(dates)
        compact = []
        for i, pred in enumerate(preds):
            # print(f'Teste pred: {pred.tolist()[0]}, {type(pred.tolist()[0])}')
            compact.append([dates[i][0], dates[i][1], pred])

        aux_df = pd.DataFrame(compact, columns=['Sem', 'DiaSem', 'Predicted'])
        aux_df['Predicted'] = aux_df['Predicted'].astype("float32")

        pvt = pd.pivot_table(aux_df, index='DiaSem', columns='Sem', values='Predicted')
        pvt.fillna(0, inplace=True)

        sns.heatmap(pvt, cmap='RdYlGn_r')
        plt.title(f'{model["name"]}')
        plt.show()


def rank_models(results: dict, verbose=False, metric='mse'):
    ranking = []
    for key, model in results.items():
        ranking.append((model["name"], round(model[metric], 5), key))
    ranking.sort(key=lambda x: x[1])
    if verbose:
        for i, model in enumerate(ranking):
            print(f'{i+1}. {model[0]}: {metric}={model[1]}')
    return ranking

# Defines the class for storing the models stats for a specific metric (f1_score, mse, rmse, r2...)
class ValidationMetrics:
    def __init__(self, model_abvs: list, model_names: list, metric: str, goal: int):
        self.metric = metric
        self.goal = goal # 0 to minimize, 1 to maximize
        self.models = {}
        for abv, name in zip(model_abvs, model_names):
            self.models[abv] = {"full_name":name, "rank_score":0, f"hist_{metric}":[],
                                   f"max_{metric}":0, f"min_{metric}":0, f"avg_{metric}":0, f"stdev_{metric}":0}
    
    def add_to_hist(self, model: str, value: int):
        self.models[model][f'hist_{self.metric}'].append(value)

    def update(self, model: str, hist=None, max=None, min=None, avg=None, stdev=None):
        stats = ['hist', 'max', 'min', 'avg', 'stdev']
        for stat in stats:
            if eval(stat) is not None:
                self.models[model][f'{stat}_{self.metric}'] = eval(stat)
    
    def batch_update(self, results: list):
        best_model = list(self.models.keys())[results.index(min(results))]
        for abv, result in zip(self.models.keys(), results):
            self.add_to_hist(abv, result)
        self.models[best_model]['rank_score'] += 1

    def fill_stats(self):
        for abv, model in self.models.items():
            _max = max(model[f'hist_{self.metric}'])
            _min = min(model[f'hist_{self.metric}'])
            avg = stt.mean(model[f'hist_{self.metric}'])
            stdev = stt.stdev(model[f'hist_{self.metric}'])
            self.update(model=abv, max=_max,  min=_min, avg=avg, stdev=stdev)

    def rank_by(self, stat='rank_score', save=False, n_iter=''):
        rank = sorted(self.models.items(), key= lambda x: x[1][stat], reverse=True)
        if save:
            generation_file = open(data_paths.GEN_PATH, mode='a')
            generation_file.write(f'---- Model Stats and Ranking - Generation [add generation number here]\n--- Iterations: {n_iter}\n--- Date and Time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\nVariables: {features}\n')
        
        for i, model in enumerate(rank):
            print(f"#{i + 1} {model[1]['full_name']}")
            if save:
                generation_file.write(f"- #{i + 1} {model[1]['full_name']}\n")
            for _stat in model[1].keys():
                if _stat not in [f'hist_{self.metric}', 'full_name']:
                    if save:
                        generation_file.write(f'    {_stat}: {round(model[1][_stat], 4)}\n')
                    print(f'    {_stat}: {round(model[1][_stat], 4)}')
        if save:
            generation_file.write('\nNotes:\n')
            generation_file.close()

    def show_performance(self, save=False, plot=True):
        labels = list(self.models.keys())
        iter_list = [x + 1 for x in range(len(self.models[labels[0]][f'hist_{self.metric}']))]
        
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            customization.color_plot(fig, ax, 'black', 'black')

            for i, model in enumerate(self.models.items()):
                plt.plot(iter_list, model[1][f'hist_{self.metric}'], 'o-', label=labels[i])

            plt.legend()
            plt.grid(axis='x')
            plt.ylabel(self.metric)
            plt.xlabel('Iteration')
            plt.title(f'Model performances over iterations for "{self.metric}"')
            plt.show()
    

# Changes matplotlib's font colors
'''text_color = 'w'

params = {"ytick.color" : text_color,
        "xtick.color" : text_color,
        "axes.labelcolor" : text_color,
        "axes.edgecolor" : text_color,
        "text.color": text_color}
plt.rcParams.update(params)'''

# Spatio-temporal information
sampling_point = 'TJ0303' # Water sampling point
time_range = [2012, 2023] # Data time range

# PATHS
inea_path = data_paths.WQ_PATH # Water quality data path
pluvio_path = data_paths.PV_PATH # Pluviometric data path


# HYPERPARAMETERS
iter_length = 5 # Number of model train-test iterations
scaling = True # Data scaling
tts_test_size = 0.1 # Train-Test-Split percentual for the testing data

pluvio_interval = '96h' # How many hours of precipitation before sampling to consider: 01h, 04h, 24h or 96h

ann_epochs = [30, 90] # How many epochs to train the ann-based models [Basic, Deep]
ann_batch_size = 2 # How many data points per batch to use during training
ann_validation_split = 0.15 # Percentual of training data to use for validation

rfr_estimators = 64 # Number of estimatores for Random Forest based models

pluvio_merge_mode = 'patternized' # How to select date and time for pluviometric data before merging
pluvio_hour = '05:00:00' # What hour to use as reference for merging the datasets
pluvio_days_past = 0 # How many days before the water quality sampling to use the pluviometric data from

outlier_stripping = ['pH', 'Coliformes Termotolerantes (NMP/100mL)',
                    'Temperatura da Água (°C)'] # Variables to check and remove outliers from


# Label - Variable to predict
model_var = 'DBO (mg/L)'
# Features - Variables to use as predictors
features = ['pH', 'Coliformes Termotolerantes (NMP/100mL)',
            'Temperatura da Água (°C)', 'Fósforo Total (mg/L)', f'Prec. max em {pluvio_interval} (mm)',
            'OD (mg/L)']
# Variables to fecth from the Water Quality dataset
fetch_vars = [model_var, 'pH', 'pH  ', 'Coliformes Termotolerantes (NMP/100mL)',
             'Coliformes Termotolerantes  (NMP/100mL)', 'Coliformes Termotolerantes (NMP/100 mL)',
             'Temperatura da Água (°C)', 'Fósforo Total (mg/L)',
             'OD (mg/L)', 'OD (mg/L )', 'OD  (mg/L)']


def protocol_gd(save_data=False):
    # Fetches the data from the Water Quality dataset and inserts it into a dataframe.
    df = data_fetching.fetch_data(inea_path, ['Data', 'Hora'] + fetch_vars, codes=[sampling_point],
                                i_year=time_range[0], f_year=time_range[1])

    # Merges duplicate columns from the water quality dataset
    df = data_fetching.join_duplicates(df, 'OD (mg/L)', ['OD (mg/L)', 'OD (mg/L )', 'OD  (mg/L)'])
    df = data_fetching.join_duplicates(df, 'pH', ['pH', 'pH  '])
    df = data_fetching.join_duplicates(df, 'Coliformes Termotolerantes (NMP/100mL)', ['Coliformes Termotolerantes (NMP/100mL)',
                                                                                    'Coliformes Termotolerantes  (NMP/100mL)',
                                                                                    'Coliformes Termotolerantes (NMP/100 mL)'])

    # Merges the pluviometric dataset with the water quality dataset.
    df = data_fetching.merge_pluvio(df, pluvio_interval, pluvio_path, mode=pluvio_merge_mode,
                                    pattern_hour=pluvio_hour, days_past=pluvio_days_past)

    # Drops NaN values
    df.dropna(inplace=True)
    # Drops duplicated samples
    df.drop_duplicates(subset='Data', keep="last", inplace=True)

    # Removes outliers from the dataset on the variables inside the "outlier_stripping" list
    for var in (features):
        if var in outlier_stripping:
            df = data_fetching.remove_outliers(df, var)

    # Saves the DataFrame into a .csv file
    if save_data:
        df.to_csv(data_paths.DF_PATH + rf'\{sampling_point}_{time_range[0]}_{time_range[1]}.csv',
                sep=';', decimal=',', float_format='%.4f')

    print(f'Number of rows after preprocessing: {len(df)}')

    # Sorts the values by Date
    df.sort_values(by='Data', inplace=True)

    # Separates the features and labels in two lists: x and y, respectively
    x = df[features].values
    y = df[model_var].astype('float32').values

    return df, x, y


# Runs the models
def protocol_md(save=False, plot=True):
    # Models names and abberviations to initialize the ValidationMetrics() class instances
    model_abvs = ["bann", "dann", "rfr", "svr-l", "svr-p", "svr-rbf", "dtr"]
    model_names = ["Basic Neural Network", "Deep Neural Network", "Random Forest Regressor",
                "Support Vector Regressor - Linear", "Support Vector Regressor - Poly",
                "Support Vector Regressor - RBF", "Decision Tree Regressor"]

    # Instatiation of the ValidationMetrics() class
    mse_metrics = ValidationMetrics(model_abvs, model_names, 'mse', 0)
    f1_metrics = ValidationMetrics(model_abvs, model_names, 'f1', 1)

    # Beginning of the train-test iterations
    for i in range(iter_length):

        iter_res = []
        print(f'_____ Iteration n.{i + 1} _____')

        x_train, x_test, y_train, y_test = tts(x, y, test_size=0.1)
        print('TTS Done.')

        # Scales the data
        if scaling:
            scaler = MinMaxScaler().fit(x_train)
            x_train = scaler.transform(x_train)

            scaler = MinMaxScaler().fit(x_test)
            x_test = scaler.transform(x_test)
            print('Scaling Done.')


        # Models
        # Basic Neural Network
        ann_basic = ann.basic((len(x_train[0]),))
        ann_basic.fit(x_train, y_train, validation_split=ann_validation_split, epochs=ann_epochs[0],
                      batch_size=ann_batch_size, verbose=0)
        # ann_basic.save(data_paths.MS_PATH + r'\ann_basic.keras')
        ann_basic_pred = ann_basic.predict(x_test, verbose=0)
        ann_basic_mse = ann_basic.evaluate(x_test, y_test, verbose=0)
        #ann_basic_f1 = f1_score(y_test, ann_basic_pred)
        iter_res.append(ann_basic_mse)
        print('BANN Done.')

        # Deep Neural Network
        ann_deep = ann.deep((len(x_train[0]),))
        ann_deep.fit(x_train, y_train, validation_split=ann_validation_split, epochs=ann_epochs[1],
                      batch_size=ann_batch_size, verbose=0)
        # ann_deep.save(data_paths.MS_PATH + r'\ann_deep.keras')
        ann_deep_pred = ann_deep.predict(x_test, verbose=0)
        ann_deep_mse = ann_deep.evaluate(x_test, y_test, verbose=0)
        #ann_deep_f1 = f1_score(y_test, ann_deep_pred)
        iter_res.append(ann_deep_mse)
        print('DANN Done.')

        # Decision Tree Regressor
        dtr = DecisionTreeRegressor(criterion='squared_error', max_depth=len(x_train) * 2, min_samples_split=math.floor(len(df)/12))
        dtr.fit(x_train, y_train)
        dtr_pred = dtr.predict(x_test)
        dtr_mse = mse(y_test, dtr_pred)
        #dtr_f1 = f1_score(y_test, dtr_pred)
        iter_res.append(dtr_mse)
        print('DTR Done.')

        # Random Forest Regressor
        rfr = RandomForestRegressor(n_estimators=rfr_estimators, criterion='squared_error')
        rfr.fit(x_train, y_train)
        rfr_pred = rfr.predict(x_test)
        rfr_mse = mse(y_test, rfr_pred)
        #rfr_f1 = f1_score(y_test, rfr_pred)
        iter_res.append(rfr_mse)
        print('RFR Done.')

        # Support Vector Regressor - Linear
        svr_l = SVR(kernel='linear')
        svr_l.fit(x_train, y_train)
        svr_l_pred = svr_l.predict(x_test)
        svr_l_mse = mse(y_test, svr_l_pred)
        #svr_l_f1 = f1_score(y_test, svr_l_pred)
        iter_res.append(svr_l_mse)
        print('SVR-L Done.')

        # Support Vector Regressor - Poly
        svr_p = SVR(kernel='poly')
        svr_p.fit(x_train, y_train)
        svr_p_pred = svr_p.predict(x_test)
        svr_p_mse = mse(y_test, svr_p_pred)
        #svr_p_f1 = f1_score(y_test, svr_p_pred)
        iter_res.append(svr_p_mse)
        print('SVR-P Done.')

        # Support Vector Regressor - RBF
        svr_rbf = SVR(kernel='rbf')
        svr_rbf.fit(x_train, y_train)
        svr_rbf_pred = svr_rbf.predict(x_test)
        svr_rbf_mse = mse(y_test, svr_rbf_pred)
        #svr_rbf_f1 = f1_score(y_test, svr_rbf_pred)
        iter_res.append(svr_rbf_mse)
        print('SVR-RBF Done.')

        # Naive-Bayes Regressor

        # Results
        mse_metrics.batch_update(iter_res)

    mse_metrics.fill_stats()
    mse_metrics.show_performance()
    print("---- Final Ranking ----")
    mse_metrics.rank_by(save=save, n_iter=iter_length)


# Analyzes the generated data
def protocol_ex():
    axx = 'Data'
    axy = features[:]
    axy  += [model_var]
    fig, ax = plt.subplots(len(axy), figsize=((16,12)))

    for i, feature in enumerate(axy):
        print(f'Plot {i + 1} - {feature}')
        color= 'b' if (i + 1) < len(axy) else 'r'
        customization.color_plot(fig, ax[i], 'black', 'black')
        ax[i].plot(df[axx], df[feature].values, '-o', c=color)
        ax[i].set_xlabel(axx)
        ax[i].set_ylabel(feature)
        ax[i].grid(axis='y')

    plt.suptitle(f'Séries temporais {time_range[0]} - {time_range[1]}')
    plt.show()
    print(df[features + [model_var]].corr())


# Interpolates data
def protocol_in():
    itp_var = 'Temperatura da Água (°C)'
    x_ini = df['Data'].values
    # x_itp = [(data - x_ini[0] + np.timedelta64(1, 'D')).astype('timedelta64[D]') / np.timedelta64(1, 'D') for data in x_ini]
    y_itp = df[itp_var].values
    x_itp = [i+1 for i in range(len(y_itp))]

    print(x_itp)

    akima = interpolate.Akima1DInterpolator(x_itp, y_itp)
    pchip = interpolate.PchipInterpolator(x_itp, y_itp)
    cubic = interpolate.CubicSpline(x_itp, y_itp)

    x_new = np.linspace(1, 68, 544)
    interpolators = [('Akima', akima), ('Pchip', pchip), ('Cubic Spline', cubic)]

    print(f'--- Variance comparison ---\nObserved values: {round(stt.variance(y_itp), 3)}')
    for k in interpolators:
        print(f'{k[0]} interpolation: {round(stt.variance(k[1](x_new)), 3)}')

        plt.figure(figsize=((16,9)))
        plt.scatter(x_itp, y_itp, label='Observed Values')
        plt.plot(x_new, k[1](x_new), label=f'{k[0]} Interpolation', alpha=0.7, color='r')
        plt.title(f'{k[0]} Interpolation for {itp_var}')
        plt.legend()
        plt.show()
        plt.close()


# Optimizes a specific model
def protocol_op(x, y):
    dtr_ga = HPO.DTR_GA('mse', 0)
    dtr_ga.initialize_population(pop_size=400)
    dtr_ga.run_for(x=x, y=y, n_generations=50, mutation_chance=0.05, mutation_strength=5, debug=True)

# Calls the data generation protocol
df, x, y = protocol_gd()

# Starts the model train-test iterations
# protocol_md(save=False, plot=True)
# protocol_ex()