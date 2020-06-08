# -*- coding: utf-8 -*-
"""
    Miscellaneous Functions for Regression File.
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC, SVR
from sklearn.qda import QDA
import os
from sklearn.grid_search import GridSearchCV
from Neural_Network import NeuralNet

def load_dataset(path_directory, symbol): 
    """
        Import DataFrame from Dataset.
    """

    path = os.path.join(path_directory, symbol)

    out = pd.read_csv(path, index_col=2, parse_dates=[2])
    out.drop(out.columns[0], axis=1, inplace=True)

    #name = path_directory + '/sp.csv'
    #sp = pd.read_csv(name, index_col=0, parse_dates=[1])
    
    #name = path_directory + '/GOOGL.csv'
    #nasdaq = pd.read_csv(name, index_col=1, parse_dates=[1])
    
    #name = path_directory + '/treasury.csv'
    #treasury = pd.read_csv(name, index_col=0, parse_dates=[1])
    
    #return [sp, nasdaq, djia, treasury, hkong, frankfurt, paris, nikkei, london, australia]
    #return [out, nasdaq, djia, frankfurt, hkong, nikkei, australia]
    return [out]    

def count_missing(dataframe):
    """
    count number of NaN in dataframe
    """
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()

    
def addFeatures(dataframe, adjclose, returns, n):
    """
    operates on two columns of dataframe:
    - n >= 2
    - given Return_* computes the return of day i respect to day i-n. 
    - given AdjClose_* computes its moving average on n days

    """
    
    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = returns[7:] + "RolMean" + str(n)
    dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)

    exp_ma = returns[7:] + "ExponentMovingAvg" + str(n)
    dataframe[exp_ma] = pd.ewma(dataframe[returns], halflife=n)
    
def mergeDataframes(datasets):
    """
        Merge Datasets into Dataframe.
    """
    return pd.concat(datasets)

    
def applyTimeLag(dataset, lags, delta):
    """
        apply time lag to return columns selected according  to delta.
        Days to lag are contained in the lads list passed as argument.
        Returns a NaN free dataset obtained cutting the lagged dataset
        at head and tail
    """
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        newcolumn = column + str(maxLag)
        dataset[newcolumn] = dataset[column].shift(maxLag)

    return dataset.iloc[maxLag:-1, :]

# CLASSIFICATION    
def prepareDataForClassification(dataset, start_test):
    """
    generates categorical to be predicted column, attach to dataframe 
    and label the categories
    """
    le = preprocessing.LabelEncoder()
    
    dataset['UpDown'] = dataset['Return_Out']
    dataset.UpDown[dataset.UpDown >= 0] = 'Up'
    dataset.UpDown[dataset.UpDown < 0] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    
    features = dataset.columns[1:-1]
    X = dataset[features]    
    y = dataset.UpDown    
    
    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]    
    
    X_test = X[X.index >= start_test]    
    y_test = y[y.index >= start_test]
    
    return X_train, y_train, X_test, y_test    

def prepareDataForModelSelection(X_train, y_train, start_validation):
    """
    gets train set and generates a validation set splitting the train.
    The validation set is mandatory for feature and model selection.
    """
    X = X_train[X_train.index < start_validation]
    y = y_train[y_train.index < start_validation]    
    
    X_val = X_train[X_train.index >= start_validation]    
    y_val = y_train[y_train.index >= start_validation]   
    
    return X, y, X_val, y_val

  
def performClassification(X_train, y_train, X_test, y_test, method, parameters={}):
    """
        Perform Classification with the help of serveral Algorithms.
    """

    print('Performing ' + method + ' Classification...')
    print('Size of train set: ', X_train.shape)
    print('Size of test set: ', X_test.shape)
    print('Size of train set: ', y_train.shape)
    print('Size of test set: ', y_test.shape)
    

    classifiers = [
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        neighbors.KNeighborsClassifier(),
        SVC(degree=100, C=10000, epsilon=.01),
        AdaBoostRegressor(),
        AdaBoostClassifier(**parameters)(),
        GradientBoostingClassifier(n_estimators=100),
        QDA(),
    ]

    scores = []

    for classifier in classifiers:
        scores.append(benchmark_classifier(classifier, \
            X_train, y_train, X_test, y_test))

    print(scores)

def benchmark_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    #auc = roc_auc_score(y_test, clf.predict(X_test))
    return accuracy

# REGRESSION
    
def getFeatures(X_train, y_train, X_test, num_features):
    ch2 = SelectKBest(chi2, k=5)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    return X_train, X_test

def performRegression(dataset, split, symbol, output_dir):
    """
        Performing Regression on 
        Various algorithms
    """

    features = dataset.columns[1:]
    index = int(np.floor(dataset.shape[0]*split))
    train, test = dataset[:index], dataset[index:]
    print('Size of train set: ', train.shape)
    print('Size of test set: ', test.shape)
    
    #train, test = getFeatures(train[features], \
    #    train[output], test[features], 16)

    out_params = (symbol, output_dir)

    output = dataset.columns[0]

    predicted_values = []

    classifiers = [
        RandomForestRegressor(n_estimators=10, n_jobs=-1),
        SVR(C=100000, kernel='rbf', epsilon=0.1, gamma=1, degree=2),
        BaggingRegressor(),
        AdaBoostRegressor(),
        KNeighborsRegressor(),
        GradientBoostingRegressor(),
    ]

    for classifier in classifiers:

        predicted_values.append(benchmark_model(classifier, \
            train, test, features, output, out_params))

    maxiter = 1000
    batch = 150

    classifier = NeuralNet(50, learn_rate=1e-2)

    predicted_values.append(benchmark_model(classifier, \
        train, test, features, output, out_params, \
        fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9))
    

    print('-'*80)

    mean_squared_errors = []

    r2_scores = []

    for pred in predicted_values:
        mean_squared_errors.append(mean_squared_error(test[output].as_matrix(), \
            pred.as_matrix()))
        r2_scores.append(r2_score(test[output].as_matrix(), pred.as_matrix()))

    print(mean_squared_errors, r2_scores)

    return mean_squared_errors, r2_scores

def benchmark_model(model, train, test, features, output, \
    output_params, *args, **kwargs):
    '''
        Performs Training and Testing of the Data on the Model.
    '''

    print('-'*80)
    model_name = model.__str__().split('(')[0].replace('Regressor', ' Regressor')
    print(model_name)

    '''
    if 'SVR' in model.__str__():
        tuned_parameters = [{'kernel': ['rbf', 'polynomial'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        model = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % 'recall')
    '''

    symbol, output_dir = output_params

    model.fit(train[features].as_matrix(), train[output].as_matrix(), *args, **kwargs)
    predicted_value = model.predict(test[features].as_matrix())

    plt.plot(test[output].as_matrix(), color='g', ls='-', label='Actual Value')
    plt.plot(predicted_value, color='b', ls='--', label='predicted_value Value')

    plt.xlabel('Number of Set')
    plt.ylabel('Output Value')

    plt.title(model_name)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, str(symbol) + '_' \
        + model_name + '.png'), dpi=100)
    #plt.show()
    plt.clf()

    return predicted_value
