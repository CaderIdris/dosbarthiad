from typing import Any, Union

import numpy as np
import pandas as pd
import scipy as sp


def _generate_data(size: int = 5000) -> pd.DataFrame:
    """
    Generates sample data to use for the MS4H03 coursework.
    5000 rows are generated.

    Columns:
    - Diagnosis
        The dependent variable. Integer. Roughly 50:50 split between 0 and 1,
        acting as a boolean variable to determine whether the individual has a
        positive diagnosis
    - Anhedonia
        Independent variable.
            Mean: 6.50
            std: 1.49
            Skew: 0.06
            Kurt: 0.00
    - Apathy
        Independent variable.
            Mean: 2.48
            std: 1.73
            Skew: 0.15
            Kurt: -0.07
    - Appetite
        Independent variable.
            Mean: 27.07
            std: 14.2
            Skew: 0.85
            Kurt: 1.13
    - Concentration
        Independent variable.
            Mean: 6.52
            std: 1.47
            Skew: 0.01
            Kurt: -0.08
    - Content
        Independent variable.
            Mean: 0.28
            std: 0.83
            Skew: 10.55
            Kurt: 181.31
    - Delusion
        Independent variable.
            Mean: 2.64
            std: 1.44
            Skew: 0.27
            Kurt: -0.05
    - Dep_Mood
        Independent variable.
            Mean: 7.30
            std: 1.52
            Skew: -0.07
            Kurt: -0.65
    - Focus
        Independent variable.
            Mean: 6.52
            std: 1.47
            Skew: 0.01
            Kurt: -0.08
    - Hallucination
        Independent variable.
            Mean: 65.07
            std: 223.94
            Skew: 11.74
            Kurt: 210.92
    - Intrusive_Thoughts
        Independent variable.
            Mean: 5.7
            std: 2.37
            Skew: 0.13
            Kurt: -0.42
            nans: 830
    - Participant
        Always 1
    - Passive
        Independent variable.
            Mean: -421.77
            std: 496.31
            Skew: -0.3
            Kurt: -1.91
    - Pregnant
        Categorical variable.
            0 for not pregnant
            1 for pregnant
            All male participants are NaN
            nans: 2238
    - Psychomotor
        Independent variable.
            Mean: 4.68
            std: 1.48
            Skew: -0.05
            Kurt: -0.01
    - Rumination
        Independent variable.
            Mean: 5.69
            std: 2.16
            Skew: 0.19
            Kurt: -0.62
    - Sleep
        Independent variable.
            Mean: 7.01
            std: 1.41
            Skew: -0.03
            Kurt: -0.1
    - Stress
        Independent variable.
            Mean: 4.92
            std: 2.22
            Skew: -0.35
            Kurt: -0.22
    - Suspicious
        Independent variable.
            Mean: 2.75
            std: 1.5
            Skew: 0.14
            Kurt: -0.04
            nans: 2873
    - Tension
        Independent variable.
            Mean: 4.92
            std: 1.96
            Skew: -0.48
            Kurt: -0.37
    - Tired
        Independent variable.
            Mean: 5.52
            std: 1.49
            Skew: 0.08
            Kurt: -0.06
            infs: 67.0
    - Unusual_Thought
        Independent variable.
            Mean: 2.48
            std: 1.41
            Skew: 0.24
            Kurt: -0.04
    - Withdrawal
        Independent variable.
            Mean: 3.96
            std: 1.47
            Skew: -0.01
            Kurt: -0.04
    - Delay
        Categorical variable.
            Roughly 50:50 Yes/No
            Acting as boolean indicator
    - Housing
        Categorical variable.
            93.5 % Stable, 6.5 % Unstable
            Effectively boolean indicator
    - Race
        Categorical Variable
            ~1/3 White
            ~1/3 Black
            ~1/6 Hispanic
            ~1/6 Asian
    - Sex
        Categorical variable.
            ~55 % Female
            ~45 % Male
    Parameters
    ----------
    size : int, optional
        Number of rows to generate
        (Default is 5000)

    Returns
    -------
    Sample data in pd.DataFrame format
    """
    sample_data = pd.DataFrame()

    # Set a seed for the rng so the sample data remains constant
    rng = np.random.RandomState(68)

    # Set the diagnosis variables
    sample_data['Diagnosis'] = rng.randint(2, size=size)

    # Set up non categorical independent variable
    norm_dist_var_data: dict[str, dict[str, Union[float, int]]] = {
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
            'mean': 7.3, 'std': 1.52, 'skew': -0.07, 'zeros': 1075
                },
        'Focus': {
            'mean': 6.52, 'std': 1.47, 'skew': 0.01
                },
        'Intrusive_Thoughts': {
            'mean': 5.7, 'std': 2.37, 'skew': 0.13, 'nans': 830
                },
        'Passive': {
            'mean': 4.88, 'std': 1.41, 'skew': -0.02, '-999': 2125
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
            'mean': 5.52, 'std': 1.49, 'skew': 0.08, 'infs': 67
                },
        'Unusual_Thought': {
            'mean': 2.48, 'std': 1.41, 'skew': 0.24
                },
        'Withdrawal': {
            'mean': 3.96, 'std': 1.47, 'skew': -0.01}
        }
    for col, stats in norm_dist_var_data.items():
        sample_data[col] = _gen_dist(
                stats['mean'],
                stats['std'],
                stats['skew'],
                size
                )
        if stats.get('nans', 0) != 0:
            num_of_nans = round(stats['nans'] * (size / 5000))
            ind_to_replace = sample_data.loc[:, col].sample(
                    n=int(num_of_nans),
                    random_state=419
                    ).index
            sample_data.loc[ind_to_replace, col] = np.nan
        if stats.get('infs', 0) != 0:
            num_of_infs = round(stats['infs'] * (size / 5000))
            ind_to_replace = sample_data.loc[:, col].sample(
                    n=int(num_of_infs),
                    random_state=421
                    ).index
            sample_data.loc[ind_to_replace, col] = np.inf
        if stats.get('zeros', 0) != 0:
            num_of_zeros = round(stats['zeros'] * (size / 5000))
            ind_to_replace = sample_data.loc[:, col].sample(
                    n=int(num_of_zeros),
                    random_state=240
                    ).index
            sample_data.loc[ind_to_replace, col] = 0
        if stats.get('-999', 0) != 0:
            num_of_999 = round(stats['-999'] * (size / 5000))
            ind_to_replace = sample_data.loc[:, col].sample(
                    n=int(num_of_999),
                    random_state=96
                    ).index
            sample_data.loc[ind_to_replace, col] = -999

    # Add exponential column
    expon_var_data = {
            'Content': {
                'mean': 0.28, 'std': 0.83
                },
            'Hallucination': {
                'mean': 65.07, 'std': 223.94
                }
            }
    for col, stats in expon_var_data.items():
        sample_data[col] = sp.stats.expon(
                loc=stats['mean'],
                scale=stats['std'],
            ).rvs(
                size=size,
                random_state=70
                )

    # Add categorical columns
    categorical_var_dist: dict[str, dict[Union[int, str], float]] = {
        'Diagnosis': {
            1: 0.51, 0: 0.49
        },
        'Participant': {
            1: 1.0
        },
        'Delay': {
            'Yes': 0.51, 'No': 0.49
        },
        'Housing': {
            'Stable': 0.93, 'Unstable': 0.07
        },
        'Race': {
            'White': 0.37, 'Black': 0.35, 'Hispanic': 0.14, 'Asian': 0.14
        },
        'Sex': {
            'Female': 0.55, 'Male': 0.45
        }
    }
    for col, settings in categorical_var_dist.items():
        sample_data[col] = rng.choice(
                a=list(settings.keys()),
                size=size,
                p=list(settings.values())  # type: ignore
                )

    # Add pregnant column
    sample_data['Pregnant'] = np.nan
    sample_data.loc[
            sample_data.loc[:, 'Sex'] == "Female", 'Pregnant'
            ] = rng.choice(
            a=[0, 1],
            size=(sample_data['Sex'] == "Female").sum(),
            p=[0.9, 0.1]
            )

    return sample_data[
        [
            "Diagnosis",
            "Anhedonia",
            "Apathy",
            "Appetite",
            "Concentration",
            "Content",
            "Delay",
            "Delusion",
            "Dep_Mood",
            "Focus",
            "Hallucination",
            "Housing",
            "Intrusive_Thoughts",
            "Participant",
            "Passive",
            "Pregnant",
            "Psychomotor",
            "Race",
            "Rumination",
            "Sex",
            "Sleep",
            "Stress",
            "Suspicious",
            "Tension",
            "Tired",
            "Unusual_Thought",
            "Withdrawal"
        ]
    ]


def _gen_dist(mean, std, skew, size) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Generate a normal distribution with stated skew, randomly generate samples
    from the distribution of length size and scale with mean and std

    Parameters
    ----------
    mean : float
        Added to randomly generated vars after scaling
    std : float
        Scales randomly generated vars
    skew : float
        Desired skew of the distribution
    size : int
        Number of samples to generate
    """
    return sp.stats.skewnorm(
            a=skew,
            scale=std,
            loc=mean
            ).rvs(
            size=size,
            random_state=int(np.ceil(abs(std)*abs(mean)*58))
            )
