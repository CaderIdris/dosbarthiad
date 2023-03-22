from copy import deepcopy
from typing import Any, Literal, Union

import numpy as np
import pandas as pd
import sklearn.impute as imp
from sklearn.model_selection import StratifiedKFold
import sklearn.preprocessing as pp


class Clean:
    """
    Clean and standardise DataFrame for use in logistic regression

    ```

    Attributes
    ----------
    __train : pd.DataFrame
        Training data
    __test : pd.DataFrame
        Testing data

    Methods
    -------
    remove_defined(outliers)
        Remove defined values from the DataFrame and replace with nan

    remove_outliers_zscore(ignore)
        Remove outliers where zscore > 3

    scale_data(sc_type, ignore)
        Scale data based on provided method (or best performing one)

    remove_dupe_cols()
        Remove any columns where all values are duplicates

    encode(ignore=[])
        Replace categorical columns with 0 or 1

    drop_cols(cols)
        Drop specified columns

    fill(fill_type='mean', cols_to_fill=[])
        Fill missing values using specified method

    drop_nans(axis='rows')
        Drop all rows or columns with missing values

    return_df()
        Return test and train DataFrames

    """
    def __init__(self, df: pd.DataFrame, y_col: str, n_splits: int = 5):
        """
        Create an instance of the class

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to prepare

        y_col : str
            Column to stratify

        n_splits : int, optional
            Number of iterations of K Folds splitting

        """
        kfold = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=204
                )
        self.__train: dict[int, pd.DataFrame] = dict()
        self.__test: dict[int, pd.DataFrame] = dict()
        for split_index, (train, test) in enumerate(
                kfold.split(df, df.loc[:, y_col])
                ):
            self.__train[split_index] = df.iloc[train].copy()
            self.__test[split_index] = df.iloc[test].copy()

    def remove_defined(self, outliers: dict[str, list[Any]]):
        """
        Remove defined values from dataframe and replace with nan

        Parameters
        ----------
        outliers : dict
            Key is the column to replace values in, the value is a list of
            values to replace with nan
        """
        for col, outlier_vals in outliers.items():
            for outlier in outlier_vals:
                for index in self.__train.keys():
                    self.__train[index].loc[:, col] = self.__train[index].loc[
                            :, col
                            ].replace(outlier, np.nan)
                    self.__test[index].loc[:, col] = self.__test[index].loc[
                            :, col
                            ].replace(outlier, np.nan)

    def remove_outliers_zscore(
            self,
            ignore: Union[list[str], tuple[str]] = list()
            ):
        """
        Remove outliers where zscore > 3

        Parameters
        ----------
        ignore : list,tuple
            Columns to ignore
        """
        for index in self.__train.keys():
            # Extract numerical columns
            sub_train = self.__train[index].select_dtypes(include='number')
            sub_test = self.__test[index].select_dtypes(include='number')
            # Filter out unwanted columns, specified by ignore
            cols_to_examine = list(
                    filter(lambda x: x not in ignore, sub_train.columns)
                    )
            sub_train = sub_train.loc[:, cols_to_examine]
            sub_test = sub_test.loc[:, cols_to_examine]
            # Standard scale the data
            scaler = pp.StandardScaler().fit(sub_train)
            scaled_train = pd.DataFrame(
                    data=scaler.transform(sub_train),
                    columns=sub_train.columns
                    )
            scaled_test = pd.DataFrame(
                    data=scaler.transform(sub_test),
                    columns=sub_test.columns
                    )
            # Replace z score > 3 with nan and scale back
            scaled_train[scaled_train.abs() > 3] = np.nan
            scaled_test[scaled_test.abs() > 3] = np.nan
            or_train = scaler.inverse_transform(scaled_train)
            or_test = scaler.inverse_transform(scaled_test)
            # Replace columns in original df
            self.__train[index].loc[:, sub_train.columns] = or_train
            self.__test[index].loc[:, sub_test.columns] = or_test

    def scale_data(
            self,
            sc_type: Literal[
                "best",
                "standard",
                "quartile",
                "log",
                "yeo-johnson",
                "box-cox",
                "min-max"
                ],
            ignore: Union[list[str], tuple[str]] = list()
            ):
        """
        Scale data using specified method or best performing ones

        Parameters
        ----------
        sc_type : list, tuple
            Which method to use. Best picks the best one for each column.

        ignore : str
            Which columns to ignore
        """
        for index in self.__train.keys():
            sub_train = self.__train[index].select_dtypes(include='number')
            sub_test = self.__test[index].select_dtypes(include='number')
            # Filter out unwanted columns, specified by ignore
            cols_to_examine = list(
                    filter(lambda x: x not in ignore, sub_train.columns)
                    )
            sub_train = sub_train.loc[:, cols_to_examine]
            sub_test = sub_test.loc[:, cols_to_examine]
            if sc_type == "quartile":
                scaled = _quartile_scale(sub_train, sub_test)
                scaled_train = scaled['train']
                scaled_test = scaled['test']
            elif sc_type == "standard":
                scaled = _standard_scale(sub_train, sub_test)
                scaled_train = scaled['train']
                scaled_test = scaled['test']
            elif sc_type == "log":
                scaled = _log_scale(sub_train, sub_test)
                scaled_train = scaled['train']
                scaled_test = scaled['test']
            elif sc_type == "yeo-johnson":
                scaled = _yj_scale(sub_train, sub_test)
                scaled_train = scaled['train']
                scaled_test = scaled['test']
            elif sc_type == "box-cox":
                scaled = _bc_scale(sub_train, sub_test)
                scaled_train = scaled['train']
                scaled_test = scaled['test']
            elif sc_type == 'min-max':
                scaled = _mm_scale(sub_train, sub_test)
                scaled_train = scaled['train']
                scaled_test = scaled['test']
            elif sc_type == "best":
                # Use all scaling methods
                scaled_dfs: dict[str, dict[str, pd.DataFrame]] = dict()
                scaled_dfs['standard'] = _standard_scale(sub_train, sub_test)
                scaled_dfs["quartile"] = _quartile_scale(sub_train, sub_test)
                scaled_dfs["log"] = _log_scale(sub_train, sub_test)
                scaled_dfs["yj"] = _yj_scale(sub_train, sub_test)
                scaled_dfs["bc"] = _bc_scale(sub_train, sub_test)
                scaled_dfs["mm"] = _mm_scale(sub_train, sub_test)
                # Get skews
                skews: pd.DataFrame = pd.DataFrame()
                for method, dfs in scaled_dfs.items():
                    skews.loc[method, sub_train.columns] = dfs[
                            'train'
                            ].skew().abs()
                # Find lowest skews, replace all others with nans
                skew_mins = skews.min()
                skews[~skews.eq(skew_mins)] = np.nan
                # Iterate over rows, if skew value is min then use that method
                # to scale data
                scaled_train = sub_train.copy(deep=True)
                scaled_test = sub_test.copy(deep=True)
                for method, row in skews.iterrows():
                    min_cols = row.dropna()
                    for min_col in min_cols.index:
                        scaled_train.loc[:, min_col] = scaled_dfs[method][
                                'train'
                                ].loc[:, min_col]
                        scaled_test.loc[:, min_col] = scaled_dfs[method][
                                'test'
                                ].loc[:, min_col]

            # Replace __df with scaled data
            self.__train[index].loc[:, sub_train.columns] = scaled_train
            self.__test[index].loc[:, sub_test.columns] = scaled_test

    def remove_dupe_cols(self):
        """
        Removes duplicate columns, keeps first one
        """
        for index in self.__train.keys():
            dupe_bool = self.__train[index].T.duplicated()
            dupes = dupe_bool[dupe_bool].index
            self.__train[index] = self.__train[index].drop(
                    dupes,
                    axis=1
                    )
            self.__test[index] = self.__test[index].drop(dupes, axis=1)

    def encode(self, ignore: Union[list[str], tuple[str]] = list()):
        """
        Encode categorical columns with 0, 1

        Parameters
        ----------
        ignore : list, tuple
            Columns to avoid encoding
        """
        for index in self.__train.keys():
            cols = self.__train[index].select_dtypes(include='object').columns
            cols_to_encode = list(
                    filter(lambda x: x not in ignore, cols)
                    )
            for col in cols_to_encode:
                train: pd.Series = self.__train[index].loc[:, [col]]
                test: pd.Series = self.__test[index].loc[:, [col]]
                num_vals = train.nunique()[col]
                if num_vals <= 2:
                    # Single column encode
                    encoder = pp.OrdinalEncoder().fit(train)
                    self.__train[index][col] = encoder.transform(train)
                    self.__test[index][col] = encoder.transform(test)
                else:
                    # One hot encode
                    encoder = pp.OneHotEncoder().fit(train)
                    enc_train = encoder.transform(train).toarray()
                    enc_test = encoder.transform(test).toarray()
                    col_names = encoder.get_feature_names_out([col])
                    self.__train[index] = self.__train[index].drop(col, axis=1)
                    self.__test[index] = self.__test[index].drop(col, axis=1)
                    self.__train[index][col_names] = enc_train
                    self.__test[index][col_names] = enc_test

    def drop_cols(self, cols: Union[str, list[str], tuple[str]]):
        """
        Drops a column or multiple columns

        Parameters
        ----------
        cols : str, list, tuple
            Column(s) to drop
        """
        for index in self.__train.keys():
            if isinstance(cols, str):
                if cols in self.__train[index].columns:
                    cols_to_drop = [cols]
                else:
                    cols_to_drop = list()
            elif isinstance(cols, (list, tuple)):
                cols_to_drop = list(
                        filter(lambda x: x in self.__train[index].columns, cols)
                        )
            self.__train[index] = self.__train[index].drop(cols_to_drop, axis=1)
            self.__test[index] = self.__test[index].drop(cols_to_drop, axis=1)

    def fill(
            self,
            fill_type: Literal[
                'mean',
                'median',
                'freq',
                'zero',
                'knn'
                ] = 'mean',
            cols_to_fill: Union[str, list[str], tuple[str]] = list()
            ):
        """
        Fills nans with specified metric

        Parameters
        ----------
        fill_type : str, optional
            How to fill the nan values
            types:
                'mean' - Fill with mean of column
                'median' - Fill with median of column
                'freq' - Fill with most frequent value
                'zero' - Fill with zeroes
                'knn' - Impute with k Nearest Neighbours
            Default is mean
        cols_to_fill : str, list, tuple, optional
            Which columns to fill
            Defaults to empty list, which fills any columns with nan present
        """
        simple_dict: dict[str, dict[str, Union[str, int]]] = {
                'mean': {
                    'strategy': 'mean'
                    },
                'median': {
                    'strategy': 'median'
                    },
                'freq': {
                    'strategy': 'most_frequent'
                    },
                'zero': {
                    'strategy': 'constant',
                    'fill_value': 0
                    },
                }
        for index in self.__train.keys():
            cols_with_nan = self.__train[index].isna().any()
            if isinstance(cols_to_fill, str):
                cols_to_fill = [cols_to_fill]
            if cols_to_fill:
                cols_to_impute = list(
                        filter(
                            lambda x: x in cols_to_fill,
                            cols_with_nan[cols_with_nan].index
                            )
                        )
            else:
                cols_to_impute = list(cols_with_nan[cols_with_nan].index)
            train_with_nan = self.__train[index].loc[:, cols_to_impute].copy()
            test_with_nan = self.__test[index].loc[:, cols_to_impute].copy()

            if fill_type in simple_dict.keys():
                imputer = imp.SimpleImputer(
                        missing_values=np.nan,
                        **simple_dict[fill_type]
                        )
                imputer.fit(train_with_nan)
                imp_train = imputer.transform(train_with_nan)
                imp_test = imputer.transform(test_with_nan)
                self.__train[index].loc[:, cols_to_impute] = imp_train
                self.__test[index].loc[:, cols_to_impute] = imp_test
            elif fill_type == 'knn':
                imputer = imp.KNNImputer(
                        missing_values=np.nan,
                        weights='distance'
                        )
                imputer.fit(train_with_nan)
                imp_train = imputer.transform(train_with_nan)
                imp_test = imputer.transform(test_with_nan)
                self.__train[index].loc[:, cols_to_impute] = imp_train
                self.__test[index].loc[:, cols_to_impute] = imp_test

    def drop_nans(
            self,
            axis: Literal[
                'rows',
                'cols'
                ] = 'rows'
            ):
        """
        Drop either all rows containing nans or all columns

        Parameters
        ----------
        axis : str, optional
            Axis to drop, default is rows
            Either 'rows' or 'columns'
        """
        for index in self.__train.keys():
            if axis == 'rows':
                to_drop_rows_tr = self.__train[index].isna().any(axis=1)
                indices_to_drop_tr = to_drop_rows_tr[to_drop_rows_tr].index
                self.__train[index] = self.__train[index].drop(
                        indices_to_drop_tr, axis=0
                        )
                to_drop_rows_te = self.__test[index].isna().any(axis=1)
                indices_to_drop_te = to_drop_rows_te[to_drop_rows_te].index
                self.__test[index] = self.__test[index].drop(
                        indices_to_drop_te, axis=0
                        )
            elif axis == 'cols':
                to_drop_columns = self.__train[index].isna().any(axis=0)
                cols_to_drop = self.__train[index][
                    to_drop_columns[to_drop_columns].index
                ].columns
                self.__train[index] = self.__train[index].drop(
                        cols_to_drop, axis=1
                        )
                self.__test[index] = self.__test[index].drop(
                        cols_to_drop, axis=1
                        )

    def return_df(self) -> dict[str, dict[int, pd.DataFrame]]:
        """
        Returns train and test data

        Returns
        -------
        dict containing two keys:
            train: Training data
            test: Testing data
        """
        return deepcopy(
                    {
                        'train': self.__train,
                        'test': self.__test
                    }
                )


def _quartile_scale(
        train: pd.DataFrame,
        test: pd.DataFrame
        ) -> dict[str, pd.DataFrame]:
    """
    Performs quartile scaling using scikit-learns RobustScaler class

    Parameters
    ----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    """
    scaler = pp.RobustScaler().fit(train)
    return {
            'train': pd.DataFrame(
                scaler.transform(train),
                columns=train.columns,
                index=train.index
                ),
            'test': pd.DataFrame(
                scaler.transform(test),
                columns=test.columns,
                index=test.index
                )
            }


def _standard_scale(
        train: pd.DataFrame,
        test: pd.DataFrame
        ) -> dict[str, pd.DataFrame]:
    """
    Performs standard scaling using scikit-learns StandardScaler class

    Parameters
    ----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    """
    scaler = pp.StandardScaler().fit(train)
    return {
            'train': pd.DataFrame(
                scaler.transform(train),
                columns=train.columns,
                index=train.index
                ),
            'test': pd.DataFrame(
                scaler.transform(test),
                columns=test.columns,
                index=test.index
                )
            }


def _log_scale(
        train: pd.DataFrame,
        test: pd.DataFrame
        ) -> dict[str, pd.DataFrame]:
    """
    Performs standard scaling on measurements that have been log transformed
    using scikit-learns StandardScaler class

    As measurements all need to be positive before scaling, the minimum value
    is subtracted from all measurements

    Parameters
    ----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    """
    log_train = np.log(train + 1 - train.min().min())
    log_test = np.log(test + 1 - test.min().min())
    scaler = pp.StandardScaler().fit(log_train)
    return {
            'train': pd.DataFrame(
                scaler.transform(log_train),
                columns=train.columns,
                index=train.index
                ),
            'test': pd.DataFrame(
                scaler.transform(log_test),
                columns=test.columns,
                index=test.index
                )
            }


def _yj_scale(
        train: pd.DataFrame,
        test: pd.DataFrame
        ) -> dict[str, pd.DataFrame]:
    """
    Performs yeo-johnson scaling using scikit-learns PowerTransformer class

    Parameters
    ----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    """
    scaler = pp.PowerTransformer(method='yeo-johnson').fit(train)
    return {
            'train': pd.DataFrame(
                scaler.transform(train),
                columns=train.columns,
                index=train.index
                ),
            'test': pd.DataFrame(
                scaler.transform(test),
                columns=test.columns,
                index=test.index
                )
            }


def _bc_scale(
        train: pd.DataFrame,
        test: pd.DataFrame
        ) -> dict[str, pd.DataFrame]:
    """
    Performs box-cox scaling on measurements using scikit-learns
    PowerTransformer class

    As measurements all need to be positive before scaling, the minimum value
    is subtracted from all measurements

    Parameters
    ----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    """
    pos_train = (train + 1 - train.min().min())
    pos_test = (test + 1 - test.min().min())
    scaler = pp.PowerTransformer(method='box-cox').fit(pos_train)
    return {
            'train': pd.DataFrame(
                scaler.transform(pos_train),
                columns=train.columns,
                index=train.index
                ),
            'test': pd.DataFrame(
                scaler.transform(pos_test),
                columns=test.columns,
                index=test.index
                )
            }


def _mm_scale(
        train: pd.DataFrame,
        test: pd.DataFrame
        ) -> dict[str, pd.DataFrame]:
    """
    Performs min-max scaling using scikit-learns MinMaxScaler class

    Parameters
    ----------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Testing data
    """
    scaler = pp.MinMaxScaler().fit(train)
    return {
            'train': pd.DataFrame(
                scaler.transform(train),
                columns=train.columns,
                index=train.index
                ),
            'test': pd.DataFrame(
                scaler.transform(test),
                columns=test.columns,
                index=test.index
                )
            }
