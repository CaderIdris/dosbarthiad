from typing import Any, Union

from factor_analyzer import FactorAnalyzer
import pandas as pd
import sklearn.decomposition as dc
from statsmodels.stats.outliers_influence  \
        import variance_inflation_factor as vif


class DimensionReduction:
    """
    Reduces the dimensionality of test and train data using principal
    component analysis, factor analysis and by removing columns with a high
    collinearity

    ```

    Attributes
    ----------
    __train : dict[int, pd.DataFrame]
        Dict of dataframes containing training data to be reduced
        dimensionally, int keys represent a different fold of k-folds
        validation
    __train : dict[int, pd.DataFrame]
        Dict of dataframes containing testing data to be reduced
        dimensionally, int keys represent a different fold of k-folds
        validation
    __train_other : dict[int, pd.DataFrame]
        Dict of dataframes containing training data to not be reduced,
        int keys represent a different fold of k-folds validation
    __test_other : dict[int, pd.DataFrame]
        Dict of dataframes containing testing data, to not be reduced,
        int keys represent a different fold of k-folds validation
    __decomposed : dict[str, str[ dict[int, pd.DataFrame]]]
        Dict of dicts dicts of dataframes. Top level str keys represent
        the technique used, second level str keys represent test and train data
        and third level int keys represent a different fold of k-folds
        validation
    __other_info : dict[str, dict[int, Any]]
        Any other info. Currently only used for factor loadings.

    Methods
    -------
    pca(key_name='Principal Component Analysis', cols_to_use=[])
        Performs principal component analysis on cols_to_use,
        retaining 95 % of variance in the data

    mclr(key_name='Multicollinearity', cols_to_use=[], vif_bound=5)
        Remove columns with high collinearity

    fa(key_name='Factor Analysis', cols_to_use=[])
        Performs factor analysis on cols_to_use, number of factors is
        calculated by finding how many have an eigenvalue > 1

    return_decomposed()
        Return data that has undergone dimensionality reduction

    return_other_info()
        Return all other info
    """
    def __init__(
            self,
            train: dict[int, pd.DataFrame],
            test: dict[int, pd.DataFrame],
            ignore: Union[list[str], tuple[str], str] = list()
            ):
        """
        Constructs the DimensionReduction object

        Parameters
        ----------
        train : dict[int, pd.DataFrame]
            Training data. Int keys represent a different fold of k-folds
            validation
        test : dict[int, pd.DataFrame]
            Testing data. Int keys represent a different fold of k-folds
            validation
        ignore : list[str], tuple[str], str, optional
            Column(s) to keep as they are
            Defaults to an empty list
        """
        self.__train: dict[int, pd.DataFrame] = dict()
        self.__test: dict[int, pd.DataFrame] = dict()

        self.__train_other: dict[int, pd.DataFrame] = dict()
        self.__test_other: dict[int, pd.DataFrame] = dict()

        self.__decomposed: dict[
                str, dict[str, dict[int, pd.DataFrame]]
                                ] = dict()
        self.__other_info: dict[str, dict[int, Any]] = dict()
        if isinstance(ignore, str):
            ignore = [ignore]
        for i in train.keys():
            cols_to_dc = list(
                    filter(
                        lambda x: x not in ignore,
                        train[i].columns
                    )
                )

            self.__train[i] = train[i].loc[:, cols_to_dc]
            self.__test[i] = test[i].loc[:, cols_to_dc]

            self.__train_other[i] = train[i].loc[:, ignore]
            self.__test_other[i] = test[i].loc[:, ignore]

    def pca(
        self,
        key_name: str = 'Principal Component Analysis',
        cols_to_use: list[str] = list()
    ):
        """
        Performs principal component analysis on cols_to_use,
        retaining 95 % of variance in the data

        Parameters
        ----------
        key_name : str, optional
            Key to store the dimensionally reduced data with
            Default is 'Principal Component Analysis'
        cols_to_use : list[str], optional
            Columns to reduce in dimensionality
            Default is an empty list, which means all columns in __train and
            __test are used
        """
        self.__decomposed[key_name] = dict()
        self.__decomposed[key_name]['train'] = dict()
        self.__decomposed[key_name]['test'] = dict()

        for i in self.__train.keys():
            if not cols_to_use:
                cols = self.__train[i].columns
            else:
                cols = [a for a in cols_to_use if a in self.__train[i].columns]
            pca = dc.PCA(n_components=0.95)
            train_df_keep = self.__train[i].drop(cols, axis=1)
            train_df_dc = self.__train[i].loc[:, cols]
            test_df_keep = self.__test[i].drop(cols, axis=1)
            test_df_dc = self.__test[i].loc[:, cols]
            pca.fit(train_df_dc)
            dc_cols = pca.get_feature_names_out()
            train_dc = pca.transform(train_df_dc)
            test_dc = pca.transform(test_df_dc)

            train_df = pd.DataFrame(
                data=train_dc,
                columns=dc_cols,
                index=train_df_dc.index
                )
            train_df.loc[
                    :, self.__train_other[i].columns
                    ] = self.__train_other[i]
            train_df.loc[
                    :, train_df_keep.columns
                    ] = train_df_keep
            test_df = pd.DataFrame(
                data=test_dc,
                columns=dc_cols,
                index=test_df_dc.index
                )
            test_df.loc[
                    :, self.__test_other[i].columns
                    ] = self.__test_other[i]
            test_df.loc[
                    :, test_df_keep.columns
                    ] = test_df_keep
            self.__decomposed[
                    key_name
                    ]['train'][i] = train_df
            self.__decomposed[
                    key_name
                    ]['test'][i] = test_df

    def mclr(
        self,
        key_name: str = 'Multicollinearity',
        cols_to_use: list[str] = list(),
        vif_bound: int = 5
    ):
        """
        Performs dimensionality reduction on cols_to_use,
        using the multicollinearity of the columns to make a decision

        Based on https://towardsdatascience.com/how-to-remove-multicollinearity
        -using-python-4da8d9d8abb2

        Parameters
        ----------
        key_name : str, optional
            Key to store the dimensionally reduced data with
            Default is 'Multicollinearity'
        cols_to_use : list[str], optional
            Columns to reduce in dimensionality
            Default is an empty list, which means all columns in __train and
            __test are used
        vif_bound : int, optional
            Any column with a VIF value higher than vif bound is discarded
            Defaults to 5
        """
        self.__decomposed[key_name] = dict()
        self.__decomposed[key_name]['train'] = dict()
        self.__decomposed[key_name]['test'] = dict()
        for i in self.__train.keys():
            if not cols_to_use:
                cols = self.__train[i].columns
            else:
                cols = [a for a in cols_to_use if a in self.__train[i].columns]
            train_df_keep = self.__train[i].drop(cols, axis=1)
            train_df_dc = self.__train[i].loc[:, cols]
            test_df_keep = self.__test[i].drop(cols, axis=1)
            test_df_dc = self.__test[i].loc[:, cols]

            vif_info = pd.Series(
                    data=[
                        vif(train_df_dc.values, ind)
                        for ind in range(train_df_dc.shape[1])
                        ],
                    index=train_df_dc.columns
            )
            collinear_cols = vif_info[vif_info > vif_bound].index
            if len(collinear_cols) == train_df_dc.shape[1]:
                train_df = self.__train_other[i]
                train_df.loc[:, train_df_keep.columns] = train_df_keep
                test_df = self.__test_other[i]
                test_df.loc[:, test_df_keep.columns] = test_df_keep
            else:
                train_df = train_df_dc.drop(collinear_cols, axis=1)
                train_df.loc[:, train_df_keep.columns] = train_df_keep
                train_df.loc[
                        :, self.__train_other[i].columns
                        ] = self.__train_other[i]
                test_df = test_df_dc.drop(collinear_cols, axis=1)
                test_df.loc[:, test_df_keep.columns] = test_df_keep
                test_df.loc[
                        :, self.__test_other[i].columns
                        ] = self.__test_other[i]

            self.__decomposed[key_name]['train'][i] = train_df
            self.__decomposed[key_name]['test'][i] = test_df

    def fa(
        self,
        key_name: str = 'Factor Analysis',
        cols_to_use: list[str] = list()
    ):
        """
        Performs factor analysis on cols_to_use to reduce dimensionality,
        number of factors decided by number of factors with eigenvalues above 1

        Based on https://www.analyticsvidhya.com/blog/2020/10/
        dimensionality-reduction-using-factor-analysis-in-python/

        Parameters
        ----------
        key_name : str, optional
            Key to store the dimensionally reduced data with
            Default is 'Factor Analysis'
        cols_to_use : list[str], optional
            Columns to reduce in dimensionality
            Default is an empty list, which means all columns in __train and
            __test are used
        """
        self.__decomposed[key_name] = dict()
        self.__decomposed[key_name]['train'] = dict()
        self.__decomposed[key_name]['test'] = dict()
        self.__other_info[key_name] = dict()
        for i in self.__train.keys():
            if not cols_to_use:
                cols = self.__train[i].columns
            else:
                cols = [a for a in cols_to_use if a in self.__train[i].columns]
            train_df_keep = self.__train[i].drop(cols, axis=1)
            train_df_dc = self.__train[i].loc[:, cols]
            test_df_keep = self.__test[i].drop(cols, axis=1)
            test_df_dc = self.__test[i].loc[:, cols]
            fa_eda = FactorAnalyzer(
                    rotation=None,
                    impute="drop",
                    n_factors=self.__train[i].shape[1]
                )
            fa_eda.fit(train_df_dc)
            eigenvalues, _ = fa_eda.get_eigenvalues()
            num_of_factors = (eigenvalues > 1).sum()

            col_names = [f'FA{ind}' for ind in range(num_of_factors)]
            fa = FactorAnalyzer(
                    rotation=None,
                    impute='drop',
                    n_factors=num_of_factors
                    )
            fa.fit(train_df_dc)
            train_dc = fa.transform(train_df_dc)
            test_dc = fa.transform(test_df_dc)

            train_df = pd.DataFrame(
                data=train_dc,
                columns=col_names,
                index=train_df_dc.index
                )
            train_df.loc[
                    :, self.__train_other[i].columns
                    ] = self.__train_other[i]
            train_df.loc[
                    :, train_df_keep.columns
                    ] = train_df_keep
            test_df = pd.DataFrame(
                data=test_dc,
                columns=col_names,
                index=test_df_dc.index
                )
            test_df.loc[
                    :, self.__test_other[i].columns
                    ] = self.__test_other[i]
            test_df.loc[
                    :, test_df_keep.columns
                    ] = test_df_keep

            self.__decomposed[key_name]['train'][i] = train_df
            self.__decomposed[key_name]['test'][i] = test_df

            self.__other_info[key_name][i] = pd.DataFrame(
                    fa.loadings_,
                    index=train_df_dc.columns
                    )

    def return_decomposed(self) -> dict[
            str, dict[str, dict[int, pd.DataFrame]]
            ]:
        """
        Return dimensionally reduced data stored in __decomposed
        """
        return self.__decomposed

    def return_other_info(self) -> dict[str, dict[int, Any]]:
        """
        Returns other info stored in __other_info
        """
        return self.__other_info
