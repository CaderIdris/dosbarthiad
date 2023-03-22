from typing import Any

import numpy as np
import pytest

from dosbarthiad import _generate_data, Clean


@pytest.fixture
def sample_data():
    return _generate_data(5000)


@pytest.mark.parametrize('n_splits', [2, 5, 10])
@pytest.mark.clean
def test_split(sample_data, n_splits):
    """
    Test if specified outliers replaced with nan
    """
    cleaner = Clean(
            sample_data,
            'Diagnosis',
            n_splits=n_splits
            )
    dfs = cleaner.return_df()
    train = dfs['train']
    test = dfs['test']
    assert train[0].shape[0] / (n_splits - 1) == test[0].shape[0]


@pytest.mark.clean
def test_remove_defined(sample_data):
    """
    Test if specified outliers replaced with nan
    """
    tests = dict()
    outliers: dict[str, list[Any]] = {
            "Dep_Mood": [0],
            "Passive": [-999],
            "Tired": [np.inf]
            }

    cleaner = Clean(
            sample_data,
            'Diagnosis'
            )
    cleaner.remove_defined(outliers)
    clean_df = cleaner.return_df()['test']
    for i in clean_df.keys():
        for col, outlier in outliers.items():
            for val in outlier:
                tests[f'{col} {val} removed {i}'] = (
                        not clean_df[i].loc[:, col].eq(val).any()
                        )

    for test, result in tests.items():
        print(f'{test}: {result}')

    assert all(tests.values())


@pytest.mark.clean
def test_remove_outliers(sample_data):
    """
    Test whether removing outliers via zscore works
    """
    tests = dict()
    cleaner = Clean(
            sample_data,
            'Diagnosis'
            )
    outliers: dict[str, list[Any]] = {
            "Dep_Mood": [0],
            "Passive": [-999],
            "Tired": [np.inf]
            }
    init_df = cleaner.return_df()['test']
    cleaner.remove_defined(outliers)
    cleaner.remove_outliers_zscore(
            ignore=["Diagnosis", "Participant", "Pregnant"],
            )
    new_df = cleaner.return_df()['test']
    for i in new_df.keys():
        print(new_df[i].isna().sum().sum())
        print(init_df[i].isna().sum().sum())
        tests[i] = init_df[i].isna().sum().sum() < new_df[i].isna().sum().sum()
    print(tests)
    assert all(tests.values())


@pytest.mark.clean
@pytest.mark.parametrize(
        "sc_type",
        ["best", "standard", "quartile", "log", "yeo-johnson", "box-cox"]
        )
def test_scale_data(sample_data, sc_type):
    """
    Test whether different scaling methods work as expected
    """
    ignore = ["Diagnosis", "Participant", "Pregnant"]
    tests = dict()
    cleaner = Clean(
            sample_data,
            'Diagnosis'
            )
    outliers: dict[str, list[Any]] = {
            "Dep_Mood": [0],
            "Passive": [-999],
            "Tired": [np.inf]
            }
    cleaner.remove_defined(outliers)
    cleaner.scale_data(
            ignore=ignore,
            sc_type=sc_type
            )
    raw_df = cleaner.return_df()['train']
    test_dfs = cleaner.return_df()['test']
    for i in raw_df.keys():
        tests[f'Right number of cols {i}'] = all(
                raw_df[i].columns == sample_data.columns
                )
        df = raw_df[i].select_dtypes(include='number').loc[
                :,
                list(
                    filter(
                        lambda x: x not in ignore,
                        raw_df[i].select_dtypes(include='number').columns
                    )
                )
            ]
        mean = df.mean()
        std = df.std()
        for col in mean.index:
            tests[f'Mean {col} {i}'] = abs(float(mean[col])) < 1
            tests[f'Std {col} {i}'] = std[col] == pytest.approx(1.0, 1)
        tests[f'Correct num of train rows {i}'] = raw_df[i].shape[0] == 4000
        tests[f'Correct num of test rows {i}'] = test_dfs[i].shape[0] == 1000

    for test, result in tests.items():
        print(f'{test}: {result}')
    if sc_type != "quartile":
        assert all(tests.values())
    else:
        assert True


@pytest.mark.clean
def test_dupe_column(sample_data):
    tests = dict()
    cleaner = Clean(
            sample_data,
            'Diagnosis'
            )
    cleaner.remove_dupe_cols()
    df = cleaner.return_df()['test']
    for i in df.keys():
        tests[f'Dupe gone and concentration stays {i}'] = (
                'Focus' not in df[i].columns and
                'Focus' in sample_data.columns and
                'Concentration' in df[i].columns
                )
        tests[f'Only one col gone {i}'] = (
                df[i].shape[1] == (sample_data.shape[1] - 1)
                )
        print(df[i].shape[1])
        print(sample_data.shape[1])
    assert all(tests.values())


@pytest.mark.clean
def test_encoding(sample_data):
    """
    Test whether encoding works
    """
    tests = dict()
    cleaner = Clean(
            sample_data,
            'Diagnosis'
            )
    original_df = cleaner.return_df()['test']
    cleaner.encode()
    df = cleaner.return_df()['test']
    for i in df.keys():
        tests[i] = all(
                [
                    original_df[i].shape[1] < df[i].shape[1],
                    all(
                        [
                            x in df[i].columns for x in [
                                "Race_White",
                                "Race_Black",
                                "Race_Asian",
                                "Race_Hispanic"
                                ]
                        ]
                        ),
                    'Race' not in df[i].columns
                    ]
                )
    assert all(tests.values())


@pytest.mark.clean
def test_drop_cols(sample_data):
    """
    Test whether columns are dropped
    """
    tests = dict()
    cleaner = Clean(
            sample_data,
            'Diagnosis'
            )
    cleaner.drop_cols(['Participant'])
    df = cleaner.return_df()['test']
    for i in df.keys():
        tests[f'Participant not in {i}'] = 'Participant' not in df[i].columns
    assert all(tests.values())


@pytest.mark.parametrize(
        'fill_type',
        ['mean', 'median', 'freq', 'zero', 'knn']
        )
@pytest.mark.clean
def test_fill(sample_data, fill_type):
    """
    Test whether nan values are filled properly
    """
    tests = dict()
    cleaner = Clean(
            sample_data,
            'Diagnosis'
            )
    cleaner.fill(fill_type=fill_type)
    df = cleaner.return_df()['test']
    for i in df.keys():
        tests[f'No nans in {i}'] = not df[i].isna().any().any()
    assert all(tests.values())


@pytest.mark.parametrize(
        'axis',
        ['rows', 'cols']
        )
def test_drop_nans(sample_data, axis):
    tests = dict()
    cleaner = Clean(
            sample_data,
            'Diagnosis'
            )
    or_df = cleaner.return_df()['test']
    cleaner.drop_nans(axis=axis)
    df = cleaner.return_df()['test']
    for i in df.keys():
        if axis == 'cols':
            tests[f'Less cols in {i}'] = (
                    df[i].shape[1] < or_df[i].shape[1]
                    and df[i].isna().sum().sum() == 0
                    )
        else:
            tests[f'Less rows in {i}'] = (
                    df[i].shape[0] < or_df[i].shape[0]
                    and df[i].isna().sum().sum() == 0
                    )
    assert all(tests.values())
