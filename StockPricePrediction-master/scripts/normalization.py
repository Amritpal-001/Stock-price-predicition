#! /usr/bin/python
'''
    Data Normalization
'''

from sklearn import preprocessing

def normalize(file_dataframe, cols):
    '''
        Data Normalization.
    '''

    for col in cols:
        preprocessing.normalize(file_dataframe[col], \
            axis=1, norm='l2', copy=False)

    return file_dataframe