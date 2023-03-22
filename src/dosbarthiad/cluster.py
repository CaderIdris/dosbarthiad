from typing import Any

import pandas as pd
import sklearn.cluster as cl
import sklearn.metrics as met


class Cluster:
    """
    Clusters data based on its features
    Effectively a very high level scikit-learn wrapper

    ```

    Attributes
    ----------
    __data : dict[int, pd.DataFrame]
        Combined test and train data, int keys represent a different fold of
        k-folds validation
    __group_tags : dict[int, pd.Series]
        Classifications of the data. Used to see if clusters match up with
        classifications. int keys represent a different fold of k-folds
        validation
    __cluster_tags : dict[int, pd.DataFrame]
        Tags representing which cluster a row corresponds to. Int keys
        represent a different fold of k=folds validation
    __scores : dict[int, pd.DataFrame]
        Scores testing whether clusters match classifications. Each column is a
        different scoring method, each row is a different clustering method

    Methods
    -------
    _skl_clust(name, cluster)
        Meta function that accepts any scikit-learn method that clusters an
        object based on its features

    affinity(name='Affinity Propagation Clustering', random_state=62, **kw)
        Performs affinity clustering using scikit-learns AffinityPropagation
        class

    agglomerative(name='Agglomerative Clustering', **kw)
        Performs agglomerative clustering using scikit-learns
        AgglomerativeClustering class

    birch(name="Birch Clustering", **kw)
        Performs birch clustering using scikit-learns Birch class

    dbscan(name='DBSCAN', **kw)
        Performs DBSCAN clustering using scikit-learns DBSCAN class

    kmeans(name='K-Means Clustering', random_state=62, **kw)
        Performs k-means clustering using scikit-learns KMeans class

    bisecting_kmeans(
            name='Bisecting K-Means Clustering', random_state=62, **kw
            )
        Performs bisecting k-means clustering using scikit-learns
        BisectingKMeans class

    minibatch_kmeans(
            name='Mini-Batch K-Means Clustering', random_state=62, **kw
            )
        Performs mini-batch k-means clustering using scikit-learns
        MiniBatchKMeans class

    mean_shift(name='Mean Shift Clustering', **kw)
        Performs mean shift clustering using scikit-learns MeanShift class

    optics(name='OPTICS', **kw)
        Performs OPTICS clustering using scikit-learns OPTICS class

    spectral(name='Spectral Clustering', **kw)
        Performs spectral clustering using scikit-learnsSpectralClustering
        class

    _calculate_scores()
        Tests how clusters match up to classifications using a variety of
        metrics

    set_clusters(clusters)
        Sets the cluster tags with predefined values

    return_clusters()
        Returns the cluster tags

    return_scores()
        Returns the scores
    """
    def __init__(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            y_col: str
            ):
        """
        Constructs the Cluster object

        Parameters
        ----------
        train : dict[int, pd.DataFrame]
            Training data. Int keys represent a different fold of k-folds
            validation
        test : dict[int, pd.DataFrame]
            Testing data. Int keys represent a different fold of k-folds
            validation
         y_col : str
            Name of the column with the actual classifications of the data
        """
        self.__data: dict[int, pd.DataFrame] = dict()
        self.__group_tags: dict[int, pd.Series] = dict()

        self.__cluster_tags: dict[int, pd.DataFrame] = dict()

        self.__scores: dict[int, pd.DataFrame] = dict()

        for i in train.keys():
            base_data = pd.concat([train[i], test[i]])
            self.__data[i] = base_data.drop([y_col], axis=1)
            self.__group_tags[i] = base_data.loc[:, y_col]

            self.__cluster_tags[i] = pd.DataFrame()

            self.__scores[i] = pd.DataFrame()

    def _skl_clust(
            self,
            name: str,
            cluster: Any
            ):
        """
        Meta function that accepts any scikit-learn class that clusters an
        object based on its features

        Parameters
        ----------
        name : str
            Name of the key to save the results under
        cluster : Any scikit-learn clustering class
            Clustering class to use
        """
        clust_dict: dict[int, pd.Series] = dict()
        for i, data in self.__data.items():
            clusters = pd.Series(
                    cluster.fit_predict(
                        data
                    ),
                    index=data.index
                    )
            if clusters.unique().shape[0] > 1:
                clust_dict[i] = clusters
        if len(clust_dict.keys()) == len(self.__data.keys()):
            for i, clusters in clust_dict.items():
                self.__cluster_tags[i][name] = clusters

    def affinity(
            self,
            name: str = "Affinity Propagation Clustering",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs affinity clustering using scikit-learns AffinityPropagation
        class

        Parameters
        ----------
        name : str, optional
            The name of the key to save the results under
            Default is 'Affinity Propagation Clustering'
        random_state : int, optional
            The seed for the random number generator
            Default is 62
        """
        self._skl_clust(
                name,
                cl.AffinityPropagation(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def agglomerative(
            self,
            name: str = "Agglomerative Clustering",
            **kwargs
            ):
        """
        Performs agglomerative clustering using scikit-learns
        AgglomerativeClustering class

        Parameters
        ----------
        name : str, optional
            The name of the key to save the results under
            Default is 'Agglomerative Clustering'
        random_state : int, optional
            The seed for the random number generator
            Default is 62
        """
        self._skl_clust(
                name,
                cl.AgglomerativeClustering(
                    n_clusters=2,
                    **kwargs
                    )
                )

    def birch(
            self,
            name: str = "Birch Clustering",
            **kwargs
            ):
        """
        Performs birch clustering using scikit-learns Birch
        class

        Parameters
        ----------
        name : str, optional
            The name of the key to save the results under
            Default is 'Birch Clustering'
        random_state : int, optional
            The seed for the random number generator
            Default is 62
        """
        self._skl_clust(
                name,
                cl.Birch(
                    n_clusters=2,
                    **kwargs
                    )
                )

    def dbscan(
            self,
            name: str = "DBSCAN",
            **kwargs
            ):
        """
        Performs dbscan clustering using scikit-learns DBSCAN class

        Parameters
        ----------
        name : str, optional
            The name of the key to save the results under
            Default is 'DBSCAN'
        random_state : int, optional
            The seed for the random number generator
            Default is 62
        """
        self._skl_clust(
                name,
                cl.DBSCAN(
                    **kwargs
                    )
                )

    def kmeans(
            self,
            name: str = "K-Means Clustering",
            random_state:  int = 62,
            **kwargs
            ):
        """
        Performs k-means clustering using scikit-learns KMeans class

        Parameters
        ----------
        name : str, optional
            The name of the key to save the results under
            Default is 'K-Means Clustering'
        random_state : int, optional
            The seed for the random number generator
            Default is 62
        """
        self._skl_clust(
                name,
                cl.KMeans(
                    random_state=random_state,
                    n_clusters=2,
                    n_init=10,
                    **kwargs
                    )
                )

    def bisecting_kmeans(
            self,
            name: str = "Bisecting K-Means Clustering",
            random_state:  int = 62,
            **kwargs
            ):
        """
        Performs bisecting k-means clustering using scikit-learns
        BisectingKMeans class

        Parameters
        ----------
        name : str, optional
            The name of the key to save the results under
            Default is 'Bisecting K-Means Clustering'
        random_state : int, optional
            The seed for the random number generator
            Default is 62
        """
        self._skl_clust(
                name,
                cl.BisectingKMeans(
                    random_state=random_state,
                    n_clusters=2,
                    n_init=10,
                    **kwargs
                    )
                )

    def minibatch_kmeans(
            self,
            name: str = "Mini-Batch K-Means Clustering",
            random_state:  int = 62,
            **kwargs
            ):
        """
        Performs mini-batch k-means clustering using scikit-learns
        MiniBatchKMeans class

        Parameters
        ----------
        name : str, optional
            The name of the key to save the results under
            Default is 'Mini-Batch K-Means Clustering'
        random_state : int, optional
            The seed for the random number generator
            Default is 62
        """
        self._skl_clust(
                name,
                cl.MiniBatchKMeans(
                    random_state=random_state,
                    n_clusters=2,
                    n_init=10,
                    **kwargs
                    )
                )

    def mean_shift(
            self,
            name: str = "Mean Shift Clustering",
            **kwargs
            ):
        """
        Performs mean shift clustering using scikit-learns MeanShift
        class

        Parameters
        ----------
        name : str, optional
            The name of the key to save the results under
            Default is 'Mean Shift Clustering'
        """
        self._skl_clust(
                name,
                cl.MeanShift(
                    **kwargs
                    )
                )

    def optics(
            self,
            name: str = "OPTICS",
            **kwargs
            ):
        """
        Performs OPTICS clustering using scikit-learns OPTICS
        class

        Parameters
        ----------
        name : str, optional
            The name of the key to save the results under
            Default is 'OPTICS'
        """
        self._skl_clust(
                name,
                cl.OPTICS(
                    **kwargs
                    )
                )

    def spectral(
            self,
            name: str = "Spectral Clustering",
            **kwargs
            ):
        """
        Performs spectral clustering using scikit-learns SpectralClustering
        class

        Parameters
        ----------
        name : str, optional
            The name of the key to save the results under
            Default is 'Spectral Clustering'
        """
        self._skl_clust(
                name,
                cl.SpectralClustering(
                    **kwargs
                    )
                )

    def __calculate_scores(self):
        """
        Assesses how the clusters match up to the classifications of each
        object using a variety of metrics
        """
        comp_tests: dict[str, Any] = {
                'Adjusted Mutual Info Score': met.adjusted_mutual_info_score,
                'Adjusted Rand Index': met.adjusted_rand_score,
                'Completeness Score': met.completeness_score,
                'Fowlkes-Mallows Index': met.fowlkes_mallows_score,
                'Homogeneity Score': met.homogeneity_score,
                'V-measure Cluster': met.v_measure_score
                }

        cl_tests: dict[str, Any] = {
                'Variance Ratio Criterion': met.calinski_harabasz_score,
                'Davies-Bouldin Score': met.davies_bouldin_score,
                'Silhouette Score': met.silhouette_score
                }

        for i in self.__data.keys():
            for technique, groupings in self.__cluster_tags[i].items():
                for test, method in comp_tests.items():
                    self.__scores[i].loc[technique, test] = method(
                            self.__group_tags[i], groupings
                            )
                for test, method in cl_tests.items():
                    self.__scores[i].loc[technique, test] = method(
                            self.__data[i], groupings
                            )
                confusion_matrix = met.cluster.pair_confusion_matrix(
                        self.__group_tags[i],
                        groupings
                        )
                self.__scores[i].loc[
                        technique,
                        'Confusion Matrix (True Negatives)'
                        ] = confusion_matrix[0][0]
                self.__scores[i].loc[
                        technique,
                        'Confusion Matrix (False Negatives)'
                        ] = confusion_matrix[1][0]
                self.__scores[i].loc[
                        technique,
                        'Confusion Matrix (False Positives)'
                        ] = confusion_matrix[0][1]
                self.__scores[i].loc[
                        technique,
                        'Confusion Matrix (True Positives)'
                        ] = confusion_matrix[1][1]

    def set_clusters(self, clusters: dict[int, pd.DataFrame]):
        """
        Sets the cluster tags with predefined values

        Parameters
        ----------
        clusters : dict[int, pd.DataFrame]
            Predefined cluster tags. Int keys represent a different fold of
            k-folds cross validation
        """
        self.__cluster_tags = clusters

    def return_clusters(self) -> dict[int, pd.DataFrame]:
        """
        Returns the cluster tags
        """
        return self.__cluster_tags

    def return_scores(self) -> dict[int, pd.DataFrame]:
        """
        Scores the cluster tags based on the classification values and
        returns them
        """
        self.__calculate_scores()
        return self.__scores
