import numpy as np
from scipy.stats import shapiro, normaltest

def calculate_stats(series):
    desc = series.describe(percentiles=[0.25, 0.75])
    mode = series.mode().values  # Mode can return multiple values
    mode_val = mode[0] if len(mode) > 0 else np.nan  # Take first mode or NaN
    iqr = desc['75%'] - desc['25%']
    outlier_count = ((series < desc['25%'] - 1.5*iqr) | (series > desc['75%'] + 1.5*iqr)).sum()
    normality_p = shapiro(series).pvalue if len(series) < 5000 else normaltest(series).pvalue
    stats_dict = {
        'Mean': desc['mean'],
        'Median': desc['50%'],
        'Mode': mode_val,
        'Standard_Deviation': desc['std'],
        'Variance': desc['std']**2,
        'Coefficience_of_Variance': desc['std'] / desc['mean'],
        'Range': desc['max'] - desc['min'],
        'Min': desc['min'],
        'Max': desc['max'],
        'Q1_(25%)': desc['25%'],
        'Q3_(75%)': desc['75%'],
        'IQR': iqr,
        'Skewness': series.skew(),
        'Kurtosis': series.kurt(),
        'P(Normality)': normality_p,
        'Count': len(series),
        'Outliers': outlier_count,
        'Outliers %': round(100 * outlier_count / len(series)),
    }
    return stats_dict