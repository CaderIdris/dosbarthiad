from functools import wraps
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dosbarthiad import Clean, DimensionReduction, Cluster, _generate_data


def skip_on(exception, reason):
    def decorator_func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exception:
                pytest.skip(reason)
        return wrapper
    return decorator_func


@pytest.fixture
def sample_data():
    """
    Sample data, randomly generated and cleaned
    """
    outliers: dict[str, list[Any]] = {
            "Dep_Mood": [0],
            "Passive": [-999]
            }
    ignore = ["Diagnosis", "Participant", "Pregnant"]
    data1 = _generate_data(100).drop('Tired', axis=1)
    data2_num = _generate_data(100).select_dtypes('number').drop(
            ['Diagnosis', 'Tired'], axis=1
            )
    np.random.seed(62)
    for name, data in data2_num.items():
        data2_num[name] = (
                data * 3 * np.random.random()
                ) + (3 * np.random.random())
    data2 = data1.copy()
    data2.loc[:, data2_num.columns] = data2_num
    data = pd.concat([data1, data2])
    cleaner = Clean(
            data,
            "Diagnosis",
            n_splits=2
            )
    cleaner.remove_defined(outliers)
    cleaner.scale_data(
            ignore=ignore,
            sc_type='best'
            )
    cleaner.drop_cols(['Participant'])
    cleaner.fill(fill_type='zero', cols_to_fill='Pregnant')
    cleaner.fill(fill_type='zero')
    cleaner.encode()
    dfs = cleaner.return_df()
    for df_tt in dfs.values():
        for df in df_tt.values():
            print(df.isna().any())
    red = DimensionReduction(
            dfs['train'],
            dfs['test'],
            ignore=['Diagnosis']
            )
    red.pca()
    dc_dfs = red.return_decomposed()
    return dc_dfs['Principal Component Analysis']


@pytest.mark.parametrize(
        'func',
        [
            Cluster.affinity,
            Cluster.agglomerative,
            Cluster.birch,
            Cluster.dbscan,
            Cluster.kmeans,
            Cluster.bisecting_kmeans,
            Cluster.minibatch_kmeans,
            Cluster.mean_shift,
            Cluster.optics,
            Cluster.spectral
            ]
        )
@pytest.mark.cluster
@skip_on(ValueError, "Only one cluster")
def test_clustering(sample_data, func):
    """
    """
    tests = dict()
    cluster = Cluster(
            sample_data['train'],
            sample_data['test'],
            y_col='Diagnosis'
            )
    func(cluster, name="A")
    clusts = cluster.return_scores()
    for i in clusts.keys():
        tests[f'Correct number of tests {i}'] = clusts[0].shape[1] in [13, 0]
        tests[f'Correct number of cals {i}'] = clusts[0].shape[0] in [1, 0]
    assert all(tests.values())


@pytest.mark.cluster
@skip_on(ValueError, "Only one cluster")
def test_all_classification(sample_data):
    tests = dict()
    cluster = Cluster(
            sample_data['train'],
            sample_data['test'],
            y_col='Diagnosis'
            )
    cluster.affinity()
    cluster.agglomerative()
    cluster.birch()
    cluster.dbscan()
    cluster.kmeans()
    cluster.bisecting_kmeans()
    cluster.minibatch_kmeans()
    cluster.mean_shift()
    cluster.optics()
    cluster.spectral()

    clusts = cluster.return_scores()

    for i in clusts.keys():
        tests[f'Correct number of tests {i}'] = clusts[0].shape[1] == 13
        tests[f'Correct number of cals {i}'] = clusts[0].shape[0] > 8
    assert all(tests.values())
