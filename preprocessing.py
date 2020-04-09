import argparse
import os
import warnings
import json

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

CITIES_DICT = {'iq': 0, 'sj': 1}


def create_json_obj(city, start_date, labels, features):
    dic_ts = {
        'start': str(start_date),
        'target': labels.tolist(),
        'cat': [CITIES_DICT[city]],
        'dynamic_feat': features.T.tolist()
    }
    print(f'JSON created for {city}: {labels.shape} / {features.T.shape}')
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
    
def impute_nan(train_data, imputer=None):  
    if imputer==None:
        imputer = IterativeImputer(max_iter=100, random_state=0)
    train_imp = imputer.fit_transform(train_data)
    features = train_imp
    return features, imputer

#from sklearn.exceptions import DataConversionWarning
#warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def print_shape(df):
    print('Data shape: {}, {} positive examples, {} negative examples'.format(df.shape, positive_examples, negative_examples))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--ARGNAME', type=ARGTYPE)
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
       
    train_features_csv = os.path.join('/opt/ml/processing/input', 'dengue_features_train.csv')
    print('Reading input data from {}'.format(train_features_csv))
    train_labels_csv = os.path.join('/opt/ml/processing/input', 'dengue_labels_train.csv')
    print('Reading input data from {}'.format(train_labels_csv))
    test_features_csv = os.path.join('/opt/ml/processing/input', 'dengue_features_test.csv')
    print('Reading input data from {}'.format(test_features_csv))
    
    train_features_df = pd.read_csv(train_features_csv)
    train_labels_df = pd.read_csv(train_labels_csv)
    test_features_df = pd.read_csv(test_features_csv)
   
    columns = ['city', 'year', 'weekofyear', 'week_start_date', 'ndvi_ne', 'ndvi_nw',
       'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm', 'total_cases']


    train_df = train_features_df.merge(train_labels_df, left_on=['city', 'year', 'weekofyear'], right_on=['city', 'year', 'weekofyear'])
      
    #preprocess = make_column_transformer(
     #   (['age', 'num persons worked for employer'], KBinsDiscretizer(encode='onehot-dense', n_bins=10)),
      #  (['capital gains', 'capital losses', 'dividends from stocks'], StandardScaler()),
       # (['education', 'major industry code', 'class of worker'], OneHotEncoder(sparse=False))
    #)
    
       
    print('Running preprocessing and feature engineering transformations')
    feature_cols = [0, 1, 2, 3] 
    d_cols = ['city', 'year', 'weekofyear', 'week_start_date', 'total_cases']
    train_data_json = []
    test_data_json = []
    for city in CITIES_DICT:
        # training data
        city_data = train_df[train_df.city==city]
        start_date = city_data.week_start_date.iloc[0]
        train_features = city_data.drop(columns=d_cols)
        labels = city_data[ 'total_cases']
        train_features_imp, imputer = impute_nan(train_features)    
        train_data_json.append( create_json_obj(city, 
                                          start_date, 
                                          labels, 
                                          train_features_imp[:, :]) )
        # testing data
        city_data_test = test_features_df[test_features_df.city==city]
        test_features = city_data_test.drop(columns=d_cols, errors='ignore') # so we don't get an error on `total_cases`
        test_features_imp, _ = impute_nan(test_features, imputer)
        test_features_imp_w_context = np.append(train_features_imp, test_features_imp, axis=0)
        test_data_json.append( create_json_obj(city, 
                                          start_date, 
                                          labels, 
                                          test_features_imp_w_context[:, :]) )
    
    train_data_json_output_path = os.path.join('/opt/ml/processing/output', 'train_pp.json')   
    test_data_json_output_path_sj = os.path.join('/opt/ml/processing/output', 'test_pp_sj.json')
    test_data_json_output_path_iq = os.path.join('/opt/ml/processing/output', 'test_pp_iq.json')

    write_json(train_data_json, train_data_json_output_path)
    write_json([test_data_json[CITIES_DICT['sj']]], test_data_json_output_path_sj)
    write_json([test_data_json[CITIES_DICT['iq']]], test_data_json_output_path_iq)
