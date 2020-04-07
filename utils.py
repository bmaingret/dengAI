# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:16:19 2020

@author: Baptiste Maingret
"""

import pandas as pd
import json

TRAIN_JSON = './data/train.json'
TEST_JSON = './data/test.json'

CONTEXT_LENGTH = 260
PREDICTION_LENGTH = 260

CITIES_DICT = {'iq': 0, 'sj': 1}

def load_data():
    data_dir = './data'
    train_features_csv = f'{data_dir}/dengue_features_train.csv'
    train_labels_csv = f'{data_dir}/dengue_labels_train.csv'
    test_features_csv = f'{data_dir}/dengue_features_test.csv'
    
    train_features_df = pd.read_csv(train_features_csv)
    train_labels_df = pd.read_csv(train_labels_csv)
    test_features_df = pd.read_csv(test_features_csv)
   
    train_df = train_features_df.merge(train_labels_df, left_on=['city', 'year', 'weekofyear'], right_on=['city', 'year', 'weekofyear'])
    
    print(f'shape train feature: {train_features_df.shape}')
    print(f'shape train labels: {train_labels_df.shape}')
    print(f'shape test feature: {test_features_df.shape}')
    print(f'shape train merge: {train_df.shape}')
    
    return train_df, test_features_df

def create_json_obj(city, start_date, timeserie):
    dic_ts = {
        'start': str(start_date),
        'target': timeserie[:, 0].tolist(),
        'cat': [CITIES_DICT[city]],
        'dynamic_feat': timeserie[:,1:].T.tolist()
    }
    
    json_ts = json.dumps(dic_ts)
    return json_ts

def write_json(json_strs, filename):
    with open(filename, 'wb') as f:
        for json_str in json_strs:
            json_line = json_str + '\n'
            json_line = json_line.encode('utf-8')
            f.write(json_line)
            print(f'Wrote {len(json_line)} chars to {filename}')
    print(f'{filename} saved')


def read_json(filename):
    data = []
    with open(filename) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    return data