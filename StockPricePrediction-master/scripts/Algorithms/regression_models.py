# -*- coding: utf-8 -*-
"""
    Run Stock-Regression Algorithms
"""
from __future__ import print_function
from regression_helpers import load_dataset, addFeatures, \
    mergeDataframes, count_missing, applyTimeLag, performRegression
import sys
import os
import pickle
import traceback

def main(dir_path, output_dir):
    '''
        Run Pipeline of processes on file one by one.
    '''

    scores = {}

    files = os.listdir(dir_path)

    maxdelta = 30

    delta = range(8, maxdelta)
    print('Delta days accounted: ', max(delta))

    for file_name in files:
        try:
            symbol = file_name.split('.')[0]
            print(symbol)
            
            datasets = load_dataset(dir_path, file_name)

            for dataset in datasets:
                columns = dataset.columns
                adjclose = columns[-2]
                returns = columns[-1]
                for dele in delta:
                    addFeatures(dataset, adjclose, returns, dele)
                dataset = dataset.iloc[max(delta):,:] # computation of returns and moving means introduces NaN which are nor removed

            finance = mergeDataframes(datasets)

            high_value = 365
            high_value = min(high_value, finance.shape[0] - 1)

            lags = range(high_value, 30)
            print('Maximum time lag applied', high_value)

            if 'symbol' in finance.columns:
                finance.drop('symbol', axis=1, inplace=True)

            print('Size of data frame: ', finance.shape)
            print('Number of NaN after merging: ', count_missing(finance))

            finance = finance.interpolate(method='time')
            print('Number of NaN after time interpolation: ', finance.shape[0]*finance.shape[1] - finance.count().sum())

            finance = finance.fillna(finance.mean())
            print('Number of NaN after mean interpolation: ', (finance.shape[0]*finance.shape[1] - finance.count().sum()))

            finance.columns = [str(col.replace('&', '_and_')) for col in finance.columns]

            #Move the Open Values behind by one dataset.
            finance.open = finance.open.shift(-1)

            print(high_value)
            finance = applyTimeLag(finance, [high_value], delta)

            print('Number of NaN after temporal shifting: ', count_missing(finance))
            print('Size of data frame after feature creation: ', finance.shape)

            mean_squared_errors, r2_scores = performRegression(finance, 0.95, \
                symbol, output_dir)

            scores[symbol] = [mean_squared_errors, r2_scores]
        except Exception, e:
            pass
            traceback.print_exc()
    
    with open(os.path.join(output_dir, 'scores.pickle'), 'wb') as handle:
        pickle.dump(scores, handle)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
