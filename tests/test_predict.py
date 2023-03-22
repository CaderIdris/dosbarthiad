from typing import Any

import numpy as np
import pytest

from dosbarthiad import Clean, DimensionReduction, Predict, _generate_data


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
    data = _generate_data(500)
    cleaner = Clean(
            data,
            "Diagnosis"
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
    dfs = cleaner.return_df()
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
            Predict.log_reg,
            Predict.log_reg_cv,
            Predict.ridge,
            Predict.ridge_cv,
            Predict.passive_aggressive,
            Predict.perceptron,
            Predict.sgd,
            Predict.knn,
            Predict.decision_tree,
            Predict.extra_tree,
            Predict.extra_tree_ensemble,
            Predict.random_forest,
            Predict.gradient_boost,
            Predict.hist_gradient_boost,
            Predict.mlp,
            Predict.svc,
            Predict.linear_svc,
            Predict.nu_svc,
            Predict.lda
            ]
        )
@pytest.mark.predict
def test_single_classification(sample_data, func):
    """
    """
    tests = dict()
    predict = Predict(
            sample_data['train'],
            sample_data['test'],
            y_col='Diagnosis'
            )
    func(predict, name='A')
    preds = predict.return_pred()
    actual = predict.return_actual()
    scores = predict.return_scores()

    for i in scores.keys():
        tests[f'Correct number of tests {i}'] = scores[i].shape[1] == 13
        tests[f'Not equal {i}'] = not preds[i]['A'].eq(actual).all()

    for test, result in tests.items():
        print(f'{test}: {result}')

    assert all(tests.values())


@pytest.mark.predict
def test_all_classification(sample_data):
    tests = dict()
    predict = Predict(
            sample_data['train'],
            sample_data['test'],
            y_col='Diagnosis'
            )
    predict.log_reg()
    predict.log_reg_cv()
    predict.ridge()
    predict.ridge_cv()
    predict.passive_aggressive()
    predict.perceptron()
    predict.sgd()
    predict.knn()
    predict.decision_tree()
    predict.extra_tree()
    predict.extra_tree_ensemble()
    predict.random_forest()
    predict.gradient_boost()
    predict.hist_gradient_boost()
    predict.mlp()
    predict.svc()
    predict.linear_svc()
    predict.nu_svc()
    predict.lda()

    preds = predict.return_pred()
    actual = predict.return_actual()
    scores = predict.return_scores()

    print(scores)

    for i in scores.keys():
        tests[f'Correct num of pred keys {i}'] = preds[i].shape[1] == 19
        tests[f'Correct num of score keys {i}'] = scores[i].shape[0] == 19
        for key, item in preds[i].items():
            tests[
                    f'Not equal to real ({key}) {i}'
                    ] = not item.eq(actual[i]).all()

    for test, result in tests.items():
        print(f'{test}: {result}')

    assert all(tests.values())
