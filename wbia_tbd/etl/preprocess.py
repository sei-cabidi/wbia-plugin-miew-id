### TODO """preprocessing scripts - filtering, subsampling, splitting"""
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder



def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def load_to_df(anno_path):
    data = load_json(anno_path)

    dfa = pd.DataFrame(data['annotations'])
    dfi = pd.DataFrame(data['images'])

    df = dfa.merge(dfi, left_on='image_id', right_on='id')

    return df

def filter_viewpoint_df(df, viewpoint_list):
    df = df[df['viewpoint'].isin(viewpoint_list)]
    return df

def filter_min_names_df(df, n_filter_min):
    df = df.groupby('name').filter(lambda g: len(g)>=n_filter_min)
    return df

def subsample_max_df(df, n_subsample_max):
    df = df.groupby('name').apply(lambda g: g.sample(frac=1, random_state=0).head(n_subsample_max)).sample(frac=1, random_state=1).reset_index(drop=True)
    return df

def convert_name_to_id(names):
    le = LabelEncoder()
    names_id = le.fit_transform(names)
    return names_id



def preprocess_data(anno_path, name_keys=['name'], convert_names_to_ids=True, viewpoint_list=None, n_filter_min=None, n_subsample_max=None):

    df = load_to_df(anno_path)

    df['name'] = df[name_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    if viewpoint_list:
        df = filter_viewpoint_df(df, viewpoint_list)
    
    if n_filter_min:
        df = filter_min_names_df(df, n_filter_min)

    if n_subsample_max:
        df = subsample_max_df(df, n_subsample_max)

    if convert_names_to_ids:
        names = df['name'].values
        names_id = convert_name_to_id(names)
        df['name'] = names_id
    return df


# def make_dataframes():
#     DATA_DIR = "data/beluga-coco-v0-full"
#     IMAGES_DIR = "data/beluga-440"

#     # anno_dir = os.path.join(DATA_DIR, "annotations") 
#     anno_dir = os.path.join(DATA_DIR, "") 
#     anno_file = lambda split: f"instances_{split}2023.json"

#     train_anno_path = os.path.join(anno_dir, anno_file("train")) 
#     val_anno_path = os.path.join(anno_dir, anno_file("val")) 
#     test_anno_path = os.path.join(anno_dir, anno_file("test")) 

#     df_train = load_to_df(train_anno_path)
#     df_val = load_to_df(val_anno_path)

#     df_train = df_train[df_train['viewpoint']=='up']
#     df_val = df_val[df_val['viewpoint']=='up']

#     ## NOTE have to safely handle this case
#     df_train['name'] = df_train['name'].astype(int)
#     df_val['name'] = df_val['name'].astype(int)

#     df_train = df_train.groupby('name').filter(lambda g: len(g)>=4)
#     df_val = df_val.groupby('name').filter(lambda g: len(g)>=2)

#     # df_train.groupby('name')['name'].count().hist()
#     # df_val.groupby('name')['name'].count().hist()

#     le = LabelEncoder()
#     df_train['name'] = le.fit_transform(df_train['name'])
#     print('generated {n_train_classes} labels for the training set'.format(n_train_classes=df_train['name'].nunique()))
#     # print(df_train['name'].max(), df_train['name'].nunique())

#     ## NOTE column filtering can be done earlier to save memory for merge
#     # df_train = df_train['name', 'file_name', 'viewpoint']
#     # df_val = df_val['name', 'file_name', 'viewpoint']


#     return df_train, df_val

if __name__ == "__main__":
    pass