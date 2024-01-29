import pandas as pd
import matplotlib.pyplot as plt
from stats import intersect_stats
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
import scipy
import numpy as np

def subsample_max_df(df, group_col='name', n_subsample_max=4, random_states=(0, 1)):
    def subsample_group(g):
        return g.sample(frac=1, random_state=random_states[0]).head(n_subsample_max)

    return df.groupby(group_col).apply(subsample_group).sample(frac=1, random_state=random_states[1]).reset_index(drop=True)

def split_dataframe(df, ratio=0.5):
    part = df.sample(frac = ratio)
    rest_part = df.drop(part.index)
    return part, rest_part

def split_df(df, train_ratio=0.7, unknown_ratio=0.5, is_val=True, group_col='name', stratify_cols='name', print_key='name_viewpoint', verbose=False, random_state=0):
    
    dfa_train = df.copy()

    assert (train_ratio > 0 and train_ratio < 1)
    assert (unknown_ratio >= 0 and unknown_ratio <= 1)

    test_val_ratio = 1 - train_ratio

    
    r1 = unknown_ratio * test_val_ratio
    r2 = (test_val_ratio - r1)/((1-r1))

    print(r1, r2)  
   
    # Stratified Group Split
    if unknown_ratio > 0:
        sgkf_n_splits = max(2, round(1/r1))
        dfa_train, dfa_new = stratified_group_split(df, group_col, stratify_cols, sgkf_n_splits, random_state)
    else:
        dfa_new = None

    # Stratified Split
    if unknown_ratio < 1:
        skf_n_splits = max(2, round(1/r2))
        dfa_train, dfa_existing = stratified_split(dfa_train, group_col, skf_n_splits, random_state)
    else:
        dfa_existing = None

    dfa_test_val = pd.concat([dfa for dfa in [dfa_existing, dfa_new] if dfa is not None])
    if not is_val:
        if verbose:
            print('Calculating stats for existing subsets')
            intersect_stats(dfa_train, dfa_existing, None, key=print_key)
            print('Calculating stats for new subsets')
            intersect_stats(dfa_train, dfa_new, None, key=print_key)
            print_div()
            print('Calculating stats for combined subsets')
            intersect_stats(dfa_train, dfa_test_val, None, key=print_key)
        return dfa_train, dfa_test_val
    

    test_names, val_names = dfa_test_val[group_col].unique()[::2], dfa_test_val[group_col].unique()[1::2]
    dfa_test, dfa_val = dfa_test_val[dfa_test_val[group_col].isin(test_names)], dfa_test_val[dfa_test_val[group_col].isin(val_names)]

    if verbose:
        print_div()
        print('Calculating stats for combined subsets')
        intersect_stats(dfa_train, dfa_test, dfa_val, key=print_key)

    return dfa_train, dfa_test, dfa_val


def split_classes_objectve(r0, w, class_counts, train_ratio, unseen_ratio):
    r1 = 1 - unseen_ratio * (1 - r0)
    i0 = int(r0*len(class_counts))
    i1 = int(r1*len(class_counts))
    train_full = np.sum(class_counts[:i0], initial=0)
    train_part = w * np.sum(class_counts[i0:i1], initial=0)
    full = np.sum(class_counts)
    return np.abs(train_ratio*full - (train_full + train_part))

def split_df(df, train_ratio=0.7, unseen_ratio=0.5, is_val=True, stratify_col='name', print_key='name_viewpoint', verbose=False):
    
    assert (train_ratio > 0 and train_ratio < 1)
    assert (unseen_ratio >= 0 and unseen_ratio <= 1)

    print("Filtering...")
    df = apply_filters(df, stratify_col, None, 2)
    
    class_counts = df[stratify_col].value_counts().sort_values(ascending=False)
    sorted_classes = class_counts.index.tolist()

    res = scipy.optimize.minimize(
        lambda x: split_classes_objectve(x[0], x[1], np.array(class_counts), train_ratio, unseen_ratio), 
        [0.5, 0.5], bounds=scipy.optimize.Bounds(lb=0, ub=1, keep_feasible=True), tol=1e-12, method='Nelder-Mead')
    r0 = res.x[0]
    r1 = 1 - unseen_ratio * (1 - r0)
    i0 = int(r0*len(class_counts))
    i1 = int(r1*len(class_counts))
    w = res.x[1]

    ratios = np.zeros(len(sorted_classes))
    ratios[:i0] = 1
    ratios[i0:i1] = w

    dfa_train, dfa_test = stratified_split(df, sorted_classes, ratios, stratify_col)
    

    if not is_val:
        if verbose:
            print('Calculating stats for combined subsets')
            intersect_stats(dfa_train, dfa_test, None, key=print_key)

        return dfa_train, dfa_test

    dfa_test_val = dfa_test
    test_names, val_names = dfa_test_val[stratify_col].unique()[::2], dfa_test_val[stratify_col].unique()[1::2]
    dfa_test, dfa_val = dfa_test_val[dfa_test_val[stratify_col].isin(test_names)], dfa_test_val[dfa_test_val[stratify_col].isin(val_names)]

    if verbose:
        print_div()
        print('Calculating stats for combined subsets')
        intersect_stats(dfa_train, dfa_test, dfa_val, key=print_key)

    return dfa_train, dfa_test, dfa_val

def simple_stratified_split(df, ratio, class_col, shuffle=True):
    classes = df[class_col].unique()
    ratios = np.ones(len(classes)) * ratio
    return stratified_split(df, classes, ratios, class_col, shuffle=shuffle)


def stratified_split(df, classes, ratios, class_col, shuffle=True):
    train_indices = np.zeros(0, np.int64)
    for c, ratio in zip(classes, ratios):
        indices = np.array((df[df[class_col]==c]).index)
        if shuffle:
            np.random.shuffle(indices)
        n = int(len(indices) * ratio)
        train_indices = np.append(train_indices, indices[:n])

    train_df = df.loc[train_indices]
    test_df = df.drop(train_indices)
    return train_df, test_df



def apply_filters(dataframe, key, max_df, min_df):
        if max_df is not None:
            dataframe = subsample_max_df(dataframe, key, max_df)
        if min_df is not None:
            dataframe = filter_min_df(dataframe, key, min_df)
        return dataframe



def stratified_group_split(df, group_col, stratify_col, n_splits, random_state):
    df_comb = df.copy()
    dfg = df_comb.groupby(group_col)[group_col].count().sort_values(ascending=False)
    df_comb['image_count'] = df_comb[group_col].map(dfg)

    X = df_comb.index.values
    y = df_comb['image_count']
    groups = df_comb[group_col]
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
        break

    dfa_train = df_comb.iloc[train_index]
    dfa_new = df_comb.iloc[test_index]
    
    return dfa_train, dfa_new


   


def filter_min_df(df, key='name', min_count=2):
    return df.groupby(key).filter(lambda g: len(g) >= min_count)

def plot_distribution(train_df, test_df, val_df):
    fig, ax = plt.subplots()
    train_df['species'].value_counts().plot(kind='bar', ax=ax, label="Train")
    test_df['species'].value_counts().plot(kind='bar', ax=ax, label="Test", color='orange')
    val_df['species'].value_counts().plot(kind='bar', ax=ax, label="Val",color='green')
    ax.legend()

    print_div()
    print('Train: ')
    print_group_stats(train_df)
    print()
    print('Test: ')
    print_group_stats(test_df)
    print()
    print('Val: ')
    print_group_stats(val_df)

def print_group_stats(df):
    df_annot_counts = df['species'].value_counts(ascending=True)
    df_name_counts = df.groupby('species')['name'].nunique()
    df_stat = pd.concat([df_annot_counts, df_name_counts], axis=1)
    print(df_stat)

def print_div():
    print("===================================")