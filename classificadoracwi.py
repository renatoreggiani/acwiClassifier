# -*- coding: utf-8 -*-


#%%
#@title


import pandas as pd
import numpy as np
import pandas_datareader as pdr
import quandl

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import plot_tree

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

key_quandl="Mw-vW_dxkPHHfjxjAQsF"

# Captura e tratamento dos dados

#%%

## Dados ACWI


url_acwi = 'https://app2.msci.com/products/service/index/indexmaster/downloadLevelData?output=INDEX_LEVELS&currency_symbol=USD&index_variant=STRD&start_date=19800101&end_date=20210930&data_frequency=DAILY&baseValue=false&index_codes=892400'

# # para captura direto da web:
# df_acwi = pd.read_excel(url_acwi, skiprows=6).dropna()
# df_acwi['Date'] = pd.to_datetime(df_acwi['Date'], errors='coerce', format='%b %d, %Y')
# df_acwi.to_csv('/content/drive/MyDrive/TCC Puc Minas/ACWI.csv', index=False)

# # para executar na maq local
df_acwi = pd.read_csv(r'C:\Users\Renato\Documents\TCC\ACWI.csv',parse_dates=['Date'])
  
# para executar no colab
# df_acwi = pd.read_csv('/content/drive/MyDrive/TCC Puc Minas/ACWI.csv',parse_dates=['Date'])

df_acwi.rename(columns={df_acwi.columns[1]: df_acwi.columns[1].split()[0].upper()}, inplace=True)
df_acwi['year_week'] = df_acwi['Date'].dt.strftime('%Y-%U')
df_acwi.dropna(inplace=True)
df_acwi = df_acwi[df_acwi['Date']>'1997-01-01'].sort_values('Date')
df_acwi = df_acwi.groupby('year_week').agg('last')
df_acwi.sort_index(inplace=True)
df_acwi['log_diff'] = np.log(df_acwi['ACWI']/df_acwi['ACWI'].shift(1))
df_acwi.dropna(subset=['log_diff'], inplace=True)
df_acwi['std'] = df_acwi['log_diff'].rolling(52).std()
df_acwi.drop(columns='ACWI', inplace=True)
df_acwi

## PMI

def pmi_us_classified(key=key_quandl):

  df = quandl.get('ISM/MAN_PMI', authtoken=key, start_date="1996-11-01").sort_index()
  df['year_week'] = df.index.strftime('%Y-%U')
  mean_3m = df['PMI'].rolling(3).mean()

  #PMI classification
  df['pmi_us_gt_50_up'] = np.where((df['PMI'] > mean_3m) & (df['PMI'] >=50), 1, 0)
  df['pmi_us_gt_50_down'] = np.where((df['PMI'] < mean_3m) & (df['PMI'] >=50), 1, 0)
  df['pmi_us_lt_50_up'] = np.where((df['PMI'] > mean_3m) & (df['PMI'] < 50), 1, 0)
  df['pmi_us_lt_50_down'] = np.where((df['PMI'] < mean_3m) & (df['PMI'] < 50), 1, 0)

  df.drop(columns=[ 'PMI'], inplace=True)
  return df[df['year_week']>='1997-00'].set_index('year_week')

df_pmi = pmi_us_classified()
df_pmi.head()

## Quandl data

def get_quandl(id_quandl, curve_diff=None, key=key_quandl):
    df = quandl.get(id_quandl, authtoken=key, start_date="1996-12-01").sort_index()
    if curve_diff:
        if len(curve_diff) == 2:
            col_name = f'{id_quandl.split("/")[1]}: {"-".join(curve_diff)}'
            df[col_name] = df[curve_diff[0]] - df[curve_diff[1]]
            df = df[[col_name]]
        else:
            raise 'Diferença deve ser calculada com 2 pontos'
            
    df['year_week'] = df.index.strftime('%Y-%U')
    df = df.groupby('year_week').agg('last')
    return df

df_quandl = pd.concat([
  get_quandl('USTREASURY/HQMYC', curve_diff=('10.0', '20.0')),
  get_quandl('USTREASURY/YIELD', curve_diff=('3 MO', '7 YR')),
  get_quandl('USTREASURY/YIELD', curve_diff=('10 YR', '20 YR')),
  get_quandl('USTREASURY/YIELD', curve_diff=('10 YR', '30 YR'))
  ], join='outer', axis=1).sort_index()

df_quandl.head()

## FRED data

series_fred = ['AAA10Y', 'BAMLH0A0HYM2EY', 'CPALTT01USM657N', 'DGS10',
 'DGS3MO', 'DTB1YR', 'DTB3', 'EMVMACROBUS', 'EMVMACROCONSUME', 'EMVMACROFININD',
 'EMVTAXESEMV', 'EPUSOVDEBT', 'FEDFUNDS', 'GEPUCURRENT', 'NCBCMDPMVCE', 'POILBREUSDM',
 'STLFSI2', 'TEDRATE', 'USD3MTD156N', 'VIXCLS', ]

df_fred = pd.concat(
    [pdr.get_data_fred(serie,  start='1996-01-01', end='2021-08-10') for serie in series_fred]
    , join='outer', axis=1).sort_index()

df_fred['year_week'] = df_fred.index.strftime('%Y-%U')
df_fred = df_fred.groupby('year_week').agg('last')

df_fred.head()

## Join das capturas

df_join = pd.concat([
  df_acwi,
  df_fred,
  df_quandl,
  df_pmi
  ], join='outer', axis=1).sort_index()
df_join.info()

# 

#%% 

df_model = df_join.copy()
df_model.drop(columns=['USD3MTD156N','DGS3MO', 'FEDFUNDS', 'DTB1YR', 'Date', 'std', 'YIELD: 10 YR-30 YR'], inplace=True)
log_diff_null = df_model['log_diff'].isnull()
df_model.ffill(inplace=True)
df_model = df_model.loc[~log_diff_null.values]

df_model['DGS10'] = df_model['DGS10'].pct_change().replace(np.inf, 0)
df_model['DTB3'] = df_model['DTB3'].pct_change().replace(np.inf, 0)

median_neg = np.quantile(df_model[df_model['log_diff'] < -0.0]['log_diff'],0.5)
df_model['category'] = np.where(df_model['log_diff'] < median_neg, 1, 0)
df_model['category'] = df_model['category'].shift(-1)

df_model.dropna(inplace=True)
df_model

y = df_model['category'].values
X = df_model.drop(columns='category').values

print('Quantidade\n0: {}\n1:  {}'.format(*np.unique(y, return_counts=True)[1]))

### Variaveis explicativas com ajuste da escala

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
pd.DataFrame(X).head()


# Modelos

#%%

## Funções e seed para os modelos

class_weight = {1: y[y == 0].size / y.size,
                0: y[y == 1].size / y.size}

SEED = 51
N_ITER = 100
N_SPLITS = 100
ss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=SEED)

def valid(model, X, y):
  ss_valid = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=666)
  scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=ss_valid, n_jobs=-1)
  results = {
      'Modelo': str(model).split('(')[0],
      'Media': scores.mean(),
      'std': scores.std(),
      'Pior': scores.min(),
      'Melhor': scores.max(),
      'Parametros': model.get_params()
  }
  return pd.DataFrame([results])

  
def hp_tunning(model, params, random_state=SEED, n_iter=N_ITER, cv=ss):
  print(f'Testando hiperparametros para {str(model).split("(")[0]}' )
  
  clf = RandomizedSearchCV(model, params, random_state=random_state, scoring='balanced_accuracy',
                           n_iter=n_iter, cv=cv, n_jobs=-1, verbose=1)
  rsearch = clf.fit(X, y)
  df_rs = pd.DataFrame(rsearch.cv_results_)
  df_rs = df_rs[[col for col in df_rs.columns if not col.startswith('split')]].sort_values('rank_test_score')
  return rsearch.best_params_, df_rs

#%% Random forest

rfc = RandomForestClassifier(random_state=SEED)
rfc.fit(X, y)


rfc = RandomForestClassifier(random_state=SEED)
df_default_rfc = valid(rfc, X, y)
df_default_rfc

### Ajuste de hiperparametros

params_rfc = {
    'n_estimators':range(70, 151, 10),
    'criterion':['gini', 'entropy'], 
    'max_depth':range(20, 41, 5), 
    'min_samples_split':range(15, 25, 2), 
    'min_samples_leaf':range(50, 61, 1), 
    'max_features':['log2', 'auto'],
    'class_weight':[class_weight]
    }

rfc = RandomForestClassifier(random_state=SEED)
best_params_rfc, df_rs_rfc = hp_tunning(rfc, params_rfc)

df_rs_rfc.head()

### Melhor Random Forest 	
# MinMax    0.519023
# StdScaler 0.519405
# RobustScaler 0.519483
# Normalizer 0.507958
# QuantileTransformer 0.517181

best_params_rfc 


rfc = RandomForestClassifier(**best_params_rfc, random_state=SEED)
df_best_rfc = valid(rfc, X, y)
df_best_rfc

rfc.fit(X, y)


df_default_rfc

#%% SVC

svc = SVC(random_state=SEED)
df_default_svc = valid(svc, X, y)
df_default_svc

### Ajuste de hiperparametros

params_svc = {
    'C': range(100, 301, 10),
    'kernel':['poly', 'sigmoid', 'rbf', 'linear', ],
    'gamma':['scale', 'auto'],
    'degree': range(1, 10, 1),
    'class_weight':[class_weight]
    }

svc = SVC(random_state=SEED)
best_params_svc, df_rs_svc = hp_tunning(svc, params_svc)

df_rs_svc.head()

### Melhor SVC 
# MinMax 0.574066
# StdScaler --
# RobustScaler --
# Normalizer 
# QuantileTransformer

best_params_svc  # {'kernel': 'poly', 'gamma': 'scale', 'degree': 5, 'decision_function_shape': 'ovo', 'C': 69}


df_best_svc = valid(SVC(**best_params_svc), X,y)
df_best_svc


df_default_svc


#%% SGDClassifier

sgd = SGDClassifier(random_state=SEED)
df_default_sgd = valid(sgd, X, y)
df_default_sgd


### Ajuste de Hiperparametros

params_sgd = {
    'loss':['hinge', 'log', 'squared_hinge', 'modified_huber',  'perceptron'],
    'penalty':['l2', 'l1', 'elasticnet'],
    'alpha':[1e-3, 1e-2, 1e-1],
    'max_iter':range(5000, 19001, 2000), 
    'n_iter_no_change': [15],
    'class_weight':[class_weight]
    }

sgd = SGDClassifier(random_state=SEED)
best_params_sgd, df_rs_sgd = hp_tunning(sgd, params_sgd)

df_rs_sgd.head()

### Melhor SGDClassifier
# MinMax            0.553428
# StdScaler         0.532614
# RobustScaler      0.526218
# Normalizer        0.555143
# QuantileTransf    0.507958

best_params_sgd

best_sgd = SGDClassifier(**best_params_sgd)
df_best_sgd = valid(best_sgd, X, y)
df_best_sgd


df_default_sgd # {'penalty': 'elasticnet', 'n_iter_no_change': 20, 'max_iter': 10000, 'loss': 'squared_hinge', 'alpha': 0.01}

best_params_sgd, df_rs_sgd = hp_tunning(SGDClassifier(), params_sgd)

df_rs_sgd.head()

### Melhor SGDClassifier

best_params_sgd # {'alpha': 0.001, 'loss': 'log', 'max_iter': 1700, 'n_iter_no_change': 11, 'penalty': 'l1'}

best_sgd = SGDClassifier(**best_params_sgd)
df_best_sgd = valid(best_sgd, X, y)
df_best_sgd


df_default_sgd

#%%

# from sklearn.feature_selection import SelectKBest, mutual_info_classif


# from sklearn.feature_selection import RFECV
# svc = SVC(kernel="linear")
# svc = SVC(**best_params_svc)

# rfecv = RFECV(estimator=svc, step=1, cv=ss,
#               scoring='balanced_accuracy',
#               min_features_to_select=1, n_jobs=-1)
# rfecv.fit(X, y)
# print("Optimal number of features : %d" % rfecv.n_features_)


# X_new = rfecv.transform(X)

# df_best_svc_rfe = valid(SVC(**best_params_svc), X_new,y)


#%%

# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import chi2


# pipe = Pipeline([('selector', SelectKBest(mutual_info_classif, k=5)),
#                  ('classifier', knn)])

# params_model = { 'classifier__' + key : value for key, value in params_knn.items() }
# params_model['classifier'] = [knn]

# params = [{'selector__k': [6, 10, 20, 24]},
#           params_model]
                
# clf = RandomizedSearchCV(pipe, params, random_state=SEED, scoring='balanced_accuracy',
#                          n_iter=10, cv=ss, n_jobs=-1, verbose=1)

# clf = clf.fit(X, y)

# clf.best_estimator_
# clf.best_score_


# df_rs = pd.DataFrame(clf.cv_results_)
# df_rs = df_rs[[col for col in df_rs.columns if not col.startswith('split')]].sort_values('rank_test_score')


for k, v in best_params_rfc.items():
    print(k+'= '+str(v))