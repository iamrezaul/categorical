import math
import pandas as pd


class CategoricalConverter:
    """
    Create a categorical converter based on supplied values.

    Parameters
    ----------
    binary (boolean) : transform binary categoricals [True]

    Attributes
    ----------
    binary_length_        : size of each categorical in binary str format
    sr_categorical_mapper : pd.Series of categorical
                            (with fit values as index)

    Examples
    --------
    >>> cat_conv = CategoricalConverter(binary=False)
    >>> cat_conv
    CategoricalConverter(binary=False)
    >>> cat_conv.fit(['T' 'K' '9' 'C' 'I'])
    >>> cat_conv.transform(pd.DataFrame(
        {'A': ['K', 'T', '9'], 'B': ['C', 'I', 'K']}))

       A  B
    0  1  3
    1  2  0
    2  4  1

    >>> cat_conv = CategoricalConverter(binary=True)
    >>> cat_conv
    CategoricalConverter(binary=True)
    >>> cat_conv.fit(['T' 'K' '9' 'C' 'I'])
    >>> cat_conv.transform(pd.DataFrame(
        {'A': ['K', 'T', '9'], 'B': ['C', 'I', 'K']}))

      A_0  A_1  A_2  B_0  B_1  B_2
    0  0    0    1    0    1    1
    1  0    1    0    0    0    0
    2  1    0    0    0    0    1
    """

    def __init__(self, binary=True):
        self.values = None
        self.binary = binary
        self.binary_length_ = None
        self.sr_categorical_mapper = None

    def fit(self, values):
        """ Create the categorical mapping

        Args: 
            values: values to be mapped to categoricals
        
        Returns:
            None. Updates `self` parameters.
        """
        # None will have categorical value of 0
        self.values = [None] + list(dict.fromkeys(values))
        # Integer categorical
        codes = list(range(len(self.values)))

        self.sr_categorical_mapper = pd.Series(codes, index=self.values)

        # convert mapper to binary
        if self.binary:
            self.binary_length_ = math.ceil(math.log2(len(codes)))
            self.sr_categorical_mapper = self.sr_categorical_mapper.apply(
                lambda v: '{0:b}'.format(v).zfill(self.binary_length_))

    def transform(self, df, cols=None, fillna=0):
        """ Transforms a dataframe to categorical dataframe
        
        Args:
            df:   the dataframe to tranform

        Kwargs:
            cols   (list) : columns of the dataframe to transform
            fillna (bool) : replacement for null
        
        Return:
            DataFrame with categorical
        """
        # if cols not supplied use all columns
        if not cols:
            cols = df.columns

        # work on a copy of the dataframe
        _df = df[cols].copy()

        df_new = pd.DataFrame(index=_df.index)       # index is important

        for col in cols:
            log.info('processing column ... {}'.format(col))
            col_categories = _df[col].map(self.sr_categorical_mapper)

            if self.binary:
                new_cols = ['{}_{}'.format(col, i) for i in range(
                    self.binary_length_)]
                df_binary = pd.DataFrame(
                    col_categories.apply(
                        lambda v: list(map(int, list(str(v))))
                        if not pd.isnull(v) else [fillna]*self.binary_length_
                        ).values.tolist(),
                    index=_df.index,                # index is important
                    columns=new_cols)
                df_new = df_new.join(
                    df_binary[df_binary.columns.difference(df_new.columns)])
            else:
                df_new.loc[:, col] = col_categories

        return df_new

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(pd.DataFrame(values))

    def __repr__(self):
        return 'CategoricalConverter(binary={})'.format(self.binary)
