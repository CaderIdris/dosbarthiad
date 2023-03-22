from typing import Any

import numpy as np
import pytest

from dosbarthiad import _generate_data, Clean, DimensionReduction


@pytest.fixture
def sample_data():
    """
    Sample data, randomly generated and cleaned
    """
    outliers: dict[str, list[Any]] = {
            "Dep_Mood": [0],
            "Passive": [-999],
            "Tired": [np.inf]
            }
    ignore = ["Diagnosis", "Participant", "Pregnant"]
    data = _generate_data(5000)
    cleaner = Clean(
            data,
            'Diagnosis'
            )
    cleaner.remove_defined(outliers)
    cleaner.scale_data(
            ignore=ignore,
            sc_type='best'
            )
    cleaner.drop_cols(['Participant'])
    cleaner.fill(fill_type='zero', cols_to_fill='Pregnant')
    cleaner.fill(fill_type='knn')
    cleaner.encode()
    return cleaner.return_df()


@pytest.mark.parametrize('vif_bound', [1, 5, 10, 25, 50])
@pytest.mark.decompose
def test_mclr(sample_data, vif_bound):
    """
    Tests multicollinearity reduction
    """
    tests = dict()
    reducer = DimensionReduction(
            sample_data['train'],
            sample_data['test'],
            ignore=['Diagnosis']
            )
    reducer.mclr(vif_bound=vif_bound)
    df = reducer.return_decomposed()['Multicollinearity']
    for i in df['test'].keys():
        sub_df = df['test'][i]
        or_df = sample_data['test'][i]
        print(sub_df.columns)
        tests[i] = (
                sub_df.shape[1] < or_df.shape[1]
                and 'Diagnosis' in sub_df.columns
                )
    assert all(tests.values())


@pytest.mark.decompose
def test_pca(sample_data):
    """
    Test principal component analysis
    """
    tests = dict()
    reducer = DimensionReduction(
            sample_data['train'],
            sample_data['test'],
            ignore=['Diagnosis']
            )
    reducer.pca()
    df = reducer.return_decomposed()['Principal Component Analysis']
    for i in df['test'].keys():
        test_df = df['test'][i]
        or_df = sample_data['test'][i]
        print(test_df.columns)
        tests[i] = (
                test_df.shape[1] < or_df.shape[1]
                and 'Diagnosis' in test_df.columns
                )
    assert all(tests.values())


@pytest.mark.decompose
def test_fa(sample_data):
    """
    Test factor analysis
    """
    tests = dict()
    reducer = DimensionReduction(
            sample_data['train'],
            sample_data['test'],
            ignore=['Diagnosis']
            )
    reducer.fa()
    df = reducer.return_decomposed()['Factor Analysis']
    for i in df['test'].keys():
        test_df = df['test'][i]
        or_df = sample_data['test'][i]
        print(test_df.columns)
        tests[i] = (
                test_df.shape[1] < or_df.shape[1]
                and 'Diagnosis' in test_df.columns
                )
    assert all(tests.values())
