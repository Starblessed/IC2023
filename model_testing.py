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
import utilities
import data_paths


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

    def rank_by(self, stat='rank_score'):
        rank = sorted(self.models.items(), key= lambda x: x[1][stat], reverse=True)
        
        for i, model in enumerate(rank):
            print(f"#{i + 1} {model[1]['full_name']}")
            for _stat in model[1].keys():
                if _stat != f'hist_{self.metric}':
                    print(f'    {_stat}: {model[1][_stat]}')

    def show_performance(self):
        labels = list(self.models.keys())
        iter_list = [x + 1 for x in range(len(self.models[labels[0]][f'hist_{self.metric}']))]
        fig, ax = plt.subplots(figsize=(8, 6))
        customization.color_plot(fig, ax, 'black', 'black')
        for i, model in enumerate(self.models.items()):
            print(model, labels, iter_list)
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

sampling_point = 'TJ0303' # Water sampling point
time_range = [2013, 2023] # Data time range

inea_path = data_paths.WQ_PATH # Water quality data path
pluvio_path = data_paths.PV_PATH # Pluviometric data path

# Label - Variable to predict
model_var = 'DBO (mg/L)'
# Features - Variables to use as predictors
features = ['pH', 'Coliformes Termotolerantes (NMP/100mL)', 'Temperatura da Água (°C)', 'Prec. max em 24h (mm)']
# Variables to fecth from the Water Quality dataset
fetch_vars = [model_var, 'pH', 'pH  ', 'Coliformes Termotolerantes (NMP/100mL)',
             'Coliformes Termotolerantes  (NMP/100mL)', 'Coliformes Termotolerantes (NMP/100 mL)',
             'Temperatura da Água (°C)']

# Fetches the data from the Water Quality dataset and inserts it into a dataframe.
df = data_fetching.fetch_data(inea_path, ['Data', 'Hora'] + fetch_vars, codes=[sampling_point],
                               i_year=time_range[0], f_year=time_range[1])

# Preprocesses the data, merging duplicate columns, combining with the Pluvio dataset,
# removing outliers, NaNs and sorting by Date
df = data_fetching.join_duplicates(df, 'pH', ['pH', 'pH  '])
df = data_fetching.join_duplicates(df, 'Coliformes Termotolerantes (NMP/100mL)', ['Coliformes Termotolerantes (NMP/100mL)',
                                                                                   'Coliformes Termotolerantes  (NMP/100mL)',
                                                                                   'Coliformes Termotolerantes (NMP/100 mL)'])
df = data_fetching.merge_pluvio(df, '24h', pluvio_path)

df.dropna(inplace=True)

for var in (features):
    df = data_fetching.remove_outliers(df, var)
print(len(df))

df.sort_values(by='Data', inplace=True)

# Separates the features and labels in two lists: x and y, respectively

x = df[features].values
y = df[model_var].astype('float32').values

# Runs the models
def protocol_md():
    # Models names and abberviations to initialize the ValidationMetrics() class instances
    model_abvs = ["bann", "dann", "rfr", "svr-l", "svr-p", "svr-rbf", "dtr"]
    model_names = ["Basic Neural Network", "Deep Neural Network", "Random Forest Regressor",
                "Support Vector Regressor - Linear", "Support Vector Regressor - Poly",
                "Support Vector Regressor - RBF", "Decision Tree Regressor"]

    # Instatiation of the ValidationMetrics() class
    mse_metrics = ValidationMetrics(model_abvs, model_names, 'mse', 0)
    f1_metrics = ValidationMetrics(model_abvs, model_names, 'f1', 1)

    iter_length = 25 # Number of iterations
    scaling = True # Data scaling

    # Beginning of the train-test iterations
    for i in range(iter_length):

        iter_res = []
        print(f'Iteration n.{i + 1}')

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
        ann_basic.fit(x_train, y_train, validation_split=0.1, epochs=30, batch_size=2, verbose=0)
        ann_basic.save(data_paths.MS_PATH + r'\ann_basic.h5')
        ann_basic_pred = ann_basic.predict(x_test, verbose=0)
        ann_basic_mse = ann_basic.evaluate(x_test, y_test, verbose=0)
        #ann_basic_f1 = f1_score(y_test, ann_basic_pred)
        iter_res.append(ann_basic_mse)
        print('BANN Done.')

        # Deep Neural Network
        ann_deep = ann.deep((len(x_train[0]),))
        ann_deep.fit(x_train, y_train, validation_split=0.1, epochs=90, batch_size=2, verbose=0)
        ann_deep.save(data_paths.MS_PATH + r'\ann_deep.h5')
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
        rfr = RandomForestRegressor(n_estimators=64, criterion='squared_error')
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
        '''
        

        iter_rank = rank_models(model_results, verbose=True, metric='f1')
        f1_metrics[iter_rank[0][2]]["rank_score"] += 1

        for model, results in model_results.items():
            f1_metrics[model]["mse_hist"].append(results["mse"])
            if i == (iter_length - 1):
                f1_metrics[model]["max_f1"] = np.max(f1_metrics[model]["hist_f1"])
                f1_metrics[model]["min_f1"] = np.min(f1_metrics[model]["hist_f1"])
                f1_metrics[model]["avg_f1"] = stt.mean(f1_metrics[model]["hist_f1"])
                f1_metrics[model]["stdev_f1"] = stt.stdev(f1_metrics[model]["hist_f1"])


    # Final ranking
    # final_rank_mse = sorted(mse_metrics.items(), key=lambda x: x[1]["rank_score"])
    # reversed(final_rank_mse)
    final_rank_f1 = sorted(f1_metrics.items(), key=lambda x: x[1]["rank_score"])
    reversed(final_rank_f1)


    mnames = [model[0] for model in final_rank_f1]
    mscores = [model[1]["rank_score"] for model in final_rank_f1]

    print('--- Model Ranking ---')
    for i, model in enumerate(final_rank_f1):

        print(f'{i + 1}. {model[0]}: score={model[1]["score"]}; min_mse={model[1]["min_mse"]}; max_mse={model[1]["max_mse"]}; avg_mse={model[1]["avg_mse"]}; mse_stdev={model[1]["mse_stdev"]}')
    '''
        mse_metrics.batch_update(iter_res)

    mse_metrics.fill_stats()
    mse_metrics.show_performance()
    print("---- Final Ranking ----")
    mse_metrics.rank_by()

    '''# Sets bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    customization.color_plot(fig, ax, 'black', 'black')

    try:
        ax.bar(mnames, mscores, color='crimson')
    except:
        ax.bar(mnames, mscores, color='r')
    ax.set_xlabel('Model')
    ax.set_ylabel('Times Ranked as Best')
    ax.grid(axis='y', color='gainsboro', linestyle='-.')
    plt.suptitle(f'Model Ranking over {iter_length} iterations')
    ax.set_title('Metrics: mse')
    plt.show()
    '''


# Analyzes the data
def protocol_ex():
    axx = 'Data'
    axy = features[:]
    axy  += [model_var]
    fig, ax = plt.subplots(len(axy), figsize=((16,12)))

    for i, feature in enumerate(axy):
        print(f'Plot {i + 1} - {feature}')
        color= 'b' if (i + 1) < len(axy) else 'r'
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

protocol_ex()