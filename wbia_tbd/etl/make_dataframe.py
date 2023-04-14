### TODO """preprocessing scripts - filtering, subsampling, splitting"""
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data/beluga-coco-v0-full"
IMAGES_DIR = "data/beluga-440"

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def make_dataframes():

    # anno_dir = os.path.join(DATA_DIR, "annotations") 
    anno_dir = os.path.join(DATA_DIR, "") 
    anno_file = lambda split: f"instances_{split}2023.json"

    train_anno_path = os.path.join(anno_dir, anno_file("train")) 
    val_anno_path = os.path.join(anno_dir, anno_file("val")) 
    test_anno_path = os.path.join(anno_dir, anno_file("test")) 

    train_data = load_json(train_anno_path)
    val_data = load_json(val_anno_path)
    test_data = load_json(test_anno_path)

    dfa_train = pd.DataFrame(train_data['annotations'])
    dfi_train = pd.DataFrame(train_data['images'])

    dfa_val = pd.DataFrame(val_data['annotations'])
    dfi_val = pd.DataFrame(val_data['images'])

    # dfa_test = pd.DataFrame(test_data['annotations'])
    # dfi_test = pd.DataFrame(test_data['images'])

    df_train = dfa_train.merge(dfi_train, left_on='image_id', right_on='id')
    df_val = dfa_val.merge(dfi_val, left_on='image_id', right_on='id')
    df_train = df_train[df_train['viewpoint']=='up']
    df_val = df_val[df_val['viewpoint']=='up']

    df_train['name'] = df_train['name'].astype(int)
    df_val['name'] = df_val['name'].astype(int)

    df_train = df_train.groupby('name_viewpoint').filter(lambda g: len(g)>=4)
    df_val = df_val.groupby('name_viewpoint').filter(lambda g: len(g)>=2)

    # df_train.groupby('name')['name'].count().hist()
    # df_val.groupby('name')['name'].count().hist()

    le = LabelEncoder()
    df_train['name'] = le.fit_transform(df_train['name'])
    print('generated {n_train_classes} labels for the training set'.format(n_train_classes=df_train['name'].nunique()))
    # print(df_train['name'].max(), df_train['name'].nunique())

    return df_train, df_val