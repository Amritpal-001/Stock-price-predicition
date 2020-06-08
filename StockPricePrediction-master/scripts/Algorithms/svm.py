#! /usr/bin/python
'''
    Running Support Vector Regression Model.
'''
from __future__ import print_function

import os
import sys
import pandas as pd
from sklearn.svm import SVR
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cross_validation import train_test_split

def convert_to_integer(dt_time):
    return 10000*dt_time.year + 1000*dt_time.month + dt_time.day


def preprocess(file_dataframe, cols=['date', 'open']):
    
    if 'date' in cols:
        file_dataframe['date'].applymap(convert_to_integer)

    X = file_dataframe['open']
    y = file_dataframe['date']

    return X, y


def svm(file_dataframe, test_size=0.2, cols=['date', 'open']):
    '''
        Run Logistic Regression
    '''

    print('Loading data...')

    if 'date' in file_dataframe:
        file_dataframe['new_col'] = pd.to_datetime(file_dataframe['date']).astype(datetime)
        #file_dataframe['date'] = pd.to_datetime(file_dataframe['date'])
        file_dataframe['new_col'].apply(lambda dt_time:10000*dt_time.year + 1000*dt_time.month + dt_time.day).astype(int)

    print(file_dataframe['new_col'])

    X = file_dataframe['open']
    y = file_dataframe['new_col']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    #svr_poly = SVR(kernel='poly', C=1e3, degree=2)

    #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

    #loo = cross_validation.LeaveOneOut(len(y_train) - 1)
    #clf = grid_search.GridSearchCV(svr_rbf, parameters)
    scores = []

    #svr_rbf.fit(X_train, y_train)
    svr_lin.fit(X_train, y_train)
    #svr_poly.fit(X_train, y_train)

    #scores.append(cross_validation.cross_val_score(svr_rbf, \
    #    X_test, y_test, scoring='mean_squared_error', cv=loo).mean())
    scores.append(cross_validation.cross_val_score(svr_lin, \
        X_test, y_test, scoring='mean_squared_error', cv=loo).mean())
    #scores.append(cross_validation.cross_val_score(svr_poly, \
    #    X_test, y_test, scoring='mean_squared_error', cv=loo).mean())
    
    return scores

def main(dir_path):
    '''
        Run Pipeline of processes on file one by one.
    '''
    files = os.listdir(dir_path)

    for file_name in files:
        print(file_name)

        file_dataframe = pd.read_csv(os.path.join(dir_path, file_name), parse_dates=[1])

        print(svm(file_dataframe, 0.2, 'high'))

        break

if __name__ == '__main__':
    main(sys.argv[1])
