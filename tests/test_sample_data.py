import math

import numpy as np
import pytest

from dosbarthiad import _generate_data


@pytest.mark.sample
@pytest.mark.parametrize('size', [1000, 5000, 10000])
def test_generate_data(size):
    tests = dict()
    df = _generate_data(size)
    description = df.describe(include='all')
    description.loc['skew', :] = df.skew(numeric_only=True)
    description.loc['zeros', :] = df.eq(0).sum()
    description.loc['infs', :] = df.eq(np.inf).sum()
    description.loc['-999', :] = df.eq(-999).sum()
    description.loc['nans', :] = df.isna().sum()

    # Test independent variables
    ind_vars = {
        'Anhedonia': {
            'mean': 6.5, 'std': 1.49, 'skew': 0.06
                },
        'Apathy': {
            'mean': 2.48, 'std': 1.73, 'skew': 0.15
                },
        'Appetite': {
            'mean': 27.07, 'std': 14.2, 'skew': 0.85
                },
        'Concentration': {
            'mean': 6.52, 'std': 1.47, 'skew': 0.01
                },
        'Delusion': {
            'mean': 2.64, 'std': 1.44, 'skew': 0.27
                },
        'Dep_Mood': {
            'mean': 5.73, 'std': 3.28, 'skew': -0.82, 'zeros': 1075
                },
        'Focus': {
            'mean': 6.52, 'std': 1.47, 'skew': 0.01
                },
        'Intrusive_Thoughts': {
            'mean': 5.7, 'std': 2.37, 'skew': 0.13, 'nans': 830
                },
        'Passive': {
            'mean': -421.77, 'std': 496.31, 'skew': -0.3, '-999': 2125
                },
        'Psychomotor': {
            'mean': 4.68, 'std': 1.48, 'skew': -0.05
                },
        'Rumination': {
            'mean': 5.69, 'std': 2.16, 'skew': 0.19
                },
        'Sleep': {
            'mean': 7.01, 'std': 1.41, 'skew': -0.03
                },
        'Stress': {
            'mean': 4.92, 'std': 2.22, 'skew': -0.35
                },
        'Suspicious': {
            'mean': 2.75, 'std': 1.5, 'skew': 0.14, 'nans': 2873
                },
        'Tension': {
            'mean': 4.92, 'std': 1.96, 'skew': -0.48
                },
        'Tired': {
            'mean': np.inf, 'std': np.nan, 'skew': np.nan, 'infs': 67
                },
        'Unusual_Thought': {
            'mean': 2.48, 'std': 1.41, 'skew': 0.24
                },
        'Withdrawal': {
            'mean': 3.96, 'std': 1.47, 'skew': -0.01}
        }

    for ind_var, stats in ind_vars.items():
        desc_col = description.loc[:, ind_var]
        tests[f'{ind_var} (Mean)'] = desc_col['mean'] == pytest.approx(
                stats['mean'], abs(stats['mean'] / 10)
                )
        if not np.isnan(stats['std']):
            tests[f'{ind_var} (std)'] = desc_col['std'] == pytest.approx(
                    stats['std'], abs(stats['std'] / 5)
                    )
        else:
            tests[f'{ind_var} (std)'] = np.isnan(desc_col['std'])
        tests[f'{ind_var} (skew)'] = (
                math.copysign(
                    1, desc_col['skew']
                    ) == math.copysign(
                            1, stats['skew']
                            ) or abs(
                                desc_col['skew']
                                ) < 0.2
                    )
        if stats.get('zeros', 0) != 0:
            tests[f'{ind_var} (zeros)'] = (
                    desc_col['zeros'] * (size/5000) == pytest.approx(
                        stats['zeros'], 50
                        )
                    )
        if stats.get('nans', 0) != 0:
            tests[f'{ind_var} (nans)'] = (
                    desc_col['nans'] * (size/5000) == pytest.approx(
                        stats['nans'], 50
                        )
                    )
        if stats.get('infs', 0) != 0:
            tests[f'{ind_var} (infs)'] = (
                    desc_col['infs'] * (size/5000) == pytest.approx(
                        stats['infs'], 50
                        )
                    )
        if stats.get('-999', 0) != 0:
            tests[f'{ind_var} (-999)'] = (
                    desc_col['-999'] * (size/5000) == pytest.approx(
                        stats['-999'], 50
                        )
                    )

    # Test dependent variable
    tests['Diagnosis (zeros)'] = description.loc[
            'zeros', 'Diagnosis'
            ] == pytest.approx(2500, 100)
    tests['Diagnosis (ones)'] = 5000 - description.loc[
            'zeros', 'Diagnosis'
            ] == pytest.approx(2500, 100)

    # Test categoricals
    tests['Pregnant (zeros)'] = description.loc[
            'zeros', 'Pregnant'
            ] == pytest.approx(2400, 100)
    tests['Pregnant (ones)'] = 2700 - description.loc[
            'zeros', 'Pregnant'
            ] == pytest.approx(250, 25)
    tests['Pregnant (nans)'] = 5000 - description.loc[
            'nans', 'Pregnant'
            ] == pytest.approx(2200, 100)

    tests['Delay (Yes)'] = df.loc[:, 'Delay'].eq("Yes").sum() == pytest.approx(
            2500, 100
            )
    tests['Delay (No)'] = df.loc[:, 'Delay'].eq("No").sum() == pytest.approx(
            2500, 100
            )

    tests['Housing (Stable)'] = df.loc[:, 'Housing'].eq(
            "Stable"
            ).sum() == pytest.approx(
            4650, 50
            )
    tests['Housing (Unstable)'] = df.loc[:, 'Housing'].eq(
            "Unstable"
            ).sum() == pytest.approx(
            350, 50
            )

    tests['Participant (1)'] = (
            df.loc[:, 'Participant'].mean() == 1
            ) and (
                    df.loc[:, 'Participant'].std() == 0
                    )

    races = {
            'White': 5000/3,
            'Black': 5000/3,
            'Asian': 5000/6,
            'Hispanic': 5000/6
            }
    for race, prop in races.items():
        tests[f'Race ({race})'] = df.loc[:, 'Race'].eq(
                race
                ).sum() == pytest.approx(
                prop, 50
                )

    tests['Sex (Male)'] = df.loc[:, 'Sex'].eq(
            "Male"
            ).sum() == pytest.approx(
            2250, 50
            )
    tests['Sex (Female)'] = df.loc[:, 'Sex'].eq(
            "Female"
            ).sum() == pytest.approx(
            2750, 50
            )

    exp_vars = {
            'Content': {
                'mean': 0.28, 'std': 0.83
                },
            'Hallucination': {
                'mean': 65.07, 'std': 223.94
                },
        }

    for exp_var, stats in exp_vars.items():
        desc_col = description.loc[:, exp_var]
        tests[f'{exp_var} (std)'] = desc_col['std'] == pytest.approx(
                stats['std'], abs(stats['std'] / 5)
                )

    for test, result in tests.items():
        print(f'{test}: {result}')
    assert all(tests.values())
