import math
import pandas as pd


def get_categorical(df, cols=None, fillna=0, binary=True):
    '''
    Returns categorical values (default binary)
    
    Args:-
        df (pd.DataFrame)      : dataframe to transform
        cols (list of strings) : columns to transform
        binary (boolean)       : transform categoricals to binary
        
    Returns:-
        df_new (pd.DataFrame)  : new dataframe with categoricals
    '''
    # if cols not supplied use all columns
    if not cols:
        cols = df.columns

    # get the unique values across the columns
    unique_values = set()
    for col in cols:
        unique_values.update(df[col].fillna(fillna).unique())
    print('Unique values: {}'.format(len(unique_values)))

    # create categorical values
    categorical = pd.Categorical(unique_values)
    categories = categorical.categories
    codes = categorical.codes

    binary_length = math.ceil(math.log2(len(codes)))
    sr_category_mapper = pd.Series(categorical.codes, index=categories)
    
    # convert mapper to binary
    if binary:
        sr_category_mapper = sr_category_mapper.apply(
            lambda v: '{0:b}'.format(v).zfill(binary_length))
    
    df_new = pd.DataFrame(index=df.index)
    for col in cols:
        print('processing column {} ... '.format(col), end='')
        col_categories = df[col].fillna(fillna).map(sr_category_mapper)
        if binary:
            cols = ['{}_{}'.format(col, i) for i in range(binary_length)]
            df_new = df_new.join(pd.DataFrame(
                col_categories.apply(lambda v: list(str(v))).values.tolist(),
                columns=cols))
        else:
            df_new.loc[:, col] = col_categories
        print('DONE')
    
    return df_new
