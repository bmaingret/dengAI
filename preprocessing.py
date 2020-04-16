import argparse
import os
import warnings
import json

import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

CITIES = {'iq': 0, 'sj': 1} # Categorical variable encoding
PREDICTION_LENGTH = {'iq': 156, 'sj': 260} # Prediction length is fixed by the test set

def create_json_obj(city, start_date, labels, features=None, category=None):
    '''Creates a JSON object that can be fed to AWS SageMaker DeepAR model
      :param city: a city identifier as one of the keys of `CITIES`
      :param start_date: a string identifying the start week of the features in the following format: '1990-04-30'
      :param labels: a Serie of labels (=total_cases)
      :param features: a DataFrame with feature columns
      :return: a JSON object as string {'start': start_date (str), 'target': labels (list), 'cat': (int), 'dynamic_feat': features (list)}
      '''    
    dic_ts = {
        "start": str(start_date),
        "target": labels.tolist()}
    if category is not None:
        dic_ts["cat"] = [category]
    if features is not None:
        dic_ts["dynamic_feat"] = features.T.tolist()
        features_shape = f'/ Features: {features.T.shape}'
        
    print(f'>> JSON created for {city}. Start date: {start_date} / Target: {labels.shape} '+ features_shape)
    json_str = json.dumps(dic_ts)
    return json_str


def write_json(json_strs, filename):
    '''Write JSON objects to the filename pathn one by line (JSON Lines format)
      :param json_strs: a list of JSON str (generated by `create_json_obj` for instance)
      :param filename: target filename with path
      '''       
    with open(filename, 'wb') as f:
        for json_str in json_strs:
            json_line = json_str + '\n'
            json_line = json_line.encode('utf-8')
            f.write(json_line)
            print(f'>>> Wrote {len(json_line)} chars to {filename}')
    print(f'>> {filename} saved')

def split_train_validation(train_df, labels, prediction_length):
    '''Split trainging set into training and validation set based on the prediction length.
      :param features: feature array
      :param labesl: labels array
      :prediction_length: the prediction length that will be used
      :return: train features, validation features, train labels, validation labels     
      '''      
    train_features = train_df[:-prediction_length].copy()
    train_labels = labels[:-prediction_length].copy()
    
    train_test_features = train_df.copy()
    train_test_labels = labels.copy()
    
    validation_features = train_df.copy()
    validation_labels = labels[:-prediction_length].copy()
    
    return train_features, train_test_features, validation_features, train_labels, train_test_labels, validation_labels

if __name__=='__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--context_length_iq', type=int, default=PREDICTION_LENGTH['iq'])
    parser.add_argument('--context_length_sj', type=int, default=PREDICTION_LENGTH['sj'])
    
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
       
    # Loading data
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
       
    print('Running preprocessing and feature engineering transformations')
    train_data_json = []
    train_test_data_json = []
    validation_data_json = []
    submission_train_data_json = []
    submission_test_data_json = []
       
    for city in CITIES:
        print(f'Data sets creation for {city}')

        # Selecting features and labels, and filtering by `city`
        _train_features = train_features_df[train_features_df.city==city]
        _test_features = test_features_df[test_features_df.city==city]
        __train_labels = train_labels_df[train_labels_df.city==city]
        _train_labels = __train_labels['total_cases'] # For the labels we take the log of it

        # start_date of the timeserie
        start_date = _train_features.week_start_date.iloc[0]

        #splitting training data
        (train_train_features, 
            train_test_features,
            train_validation_features,
            train_train_labels,
            train_test_labels,
            train_validation_labels) = split_train_validation(_train_features, _train_labels, PREDICTION_LENGTH[city])

        # setting the features/labels for the test set used for submissions
        submission_train_features = _train_features.copy()
        submission_train_labels = _train_labels.copy()
        # test_features is only the features in the future for which we want to predict
        # we need to add the training features since it used for predictions
        submission_test_features = pd.concat([submission_train_features, _test_features], axis=0)        


        # preprocessing training data      
        features = ['reanalysis_dew_point_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_specific_humidity_g_per_kg']
        features_ix = [columns.index(feat) for feat in features]

        pipe = Pipeline([
            ('impute', IterativeImputer(max_iter=100, random_state=0)),
            ('scale', MaxAbsScaler())])

        column_trans = make_column_transformer(
            (pipe, features_ix))


        train_train_features = column_trans.fit_transform(train_train_features)
        train_test_features = column_trans.fit_transform(train_test_features)
        train_validation_features = column_trans.fit_transform(train_validation_features)

        # preprocessing for submission
        submission_train_features = column_trans.fit_transform(submission_train_features)
        submission_test_features = column_trans.fit_transform(submission_test_features)

        # create our json object')
        train_data_json.append( create_json_obj(city, 
                                            start_date, 
                                            train_train_labels, 
                                            train_train_features) )

        train_test_data_json.append( create_json_obj(city, 
                                            start_date, 
                                            train_test_labels, 
                                            train_test_features) )   

        validation_data_json.append( create_json_obj(city, 
                                            start_date, 
                                            train_validation_labels, 
                                            train_validation_features) )     

        submission_train_data_json.append( create_json_obj(city, 
                                            start_date, 
                                            submission_train_labels, 
                                            submission_train_features) )

        submission_test_data_json.append( create_json_obj(city, 
                                            start_date, 
                                            submission_train_labels, 
                                            submission_test_features) )

    # json files path
    train_data_json_output_path = os.path.join('/opt/ml/processing/output', 'train_pp.json')
    train_data_json_output_path_sj = os.path.join('/opt/ml/processing/output', 'train_pp_sj.json')
    train_data_json_output_path_iq = os.path.join('/opt/ml/processing/output', 'train_pp_iq.json')

    train_test_data_json_output_path_sj = os.path.join('/opt/ml/processing/output', 'train_test_pp_sj.json')
    train_test_data_json_output_path_iq = os.path.join('/opt/ml/processing/output', 'train_test_pp_iq.json')        

    validation_data_json_output_path_sj = os.path.join('/opt/ml/processing/output', 'validation_pp_sj.json')
    validation_data_json_output_path_iq = os.path.join('/opt/ml/processing/output', 'validation_pp_iq.json')    

    submission_train_data_json_output_path_sj = os.path.join('/opt/ml/processing/output', 'submission_train_pp_sj.json')
    submission_train_data_json_output_path_iq = os.path.join('/opt/ml/processing/output', 'submission_train_pp_iq.json')

    submission_test_data_json_output_path_sj = os.path.join('/opt/ml/processing/output', 'submission_test_pp_sj.json')
    submission_test_data_json_output_path_iq = os.path.join('/opt/ml/processing/output', 'submission_test_pp_iq.json')

    #write our json files 
    write_json(train_data_json, train_data_json_output_path)
    write_json([train_data_json[CITIES['sj']]], train_data_json_output_path_sj)
    write_json([train_data_json[CITIES['iq']]], train_data_json_output_path_iq)    

    write_json([train_test_data_json[CITIES['sj']]], train_test_data_json_output_path_sj)
    write_json([train_test_data_json[CITIES['iq']]], train_test_data_json_output_path_iq)    

    write_json([validation_data_json[CITIES['sj']]], validation_data_json_output_path_sj)
    write_json([validation_data_json[CITIES['iq']]], validation_data_json_output_path_iq)    

    write_json([submission_train_data_json[CITIES['sj']]], submission_train_data_json_output_path_sj)
    write_json([submission_train_data_json[CITIES['iq']]], submission_train_data_json_output_path_iq)
    write_json([submission_test_data_json[CITIES['sj']]], submission_test_data_json_output_path_sj)
    write_json([submission_test_data_json[CITIES['iq']]], submission_test_data_json_output_path_iq)