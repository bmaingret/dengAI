# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:16:19 2020

@author: Baptiste Maingret
"""

import pandas as pd

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