#! /usr/bin/python
'''
    Main File.
'''
import os
import sys
import pandas as pd

from interpolation import interpolate
from normalization import normalize


def main(dir_path, output_dir):
    '''
        Run Pipeline of processes on file one by one.
    '''
    files = os.listdir(dir_path)

    for file_name in files:

        file_dataframe = pd.read_csv(os.path.join(dir_path, file_name))

        cols = ['high', 'open', 'low', 'close', 'volume', 'adj_close']

        file_dataframe = interpolate(file_dataframe, cols)

        file_dataframe = normalize(file_dataframe, cols)

        file_dataframe.to_csv(
            os.path.join(output_dir, file_name), encoding='utf-8')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
