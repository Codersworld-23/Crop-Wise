import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.express as px

# -- Normal Distribution & CLT --
def plot_distribution(data, column):
    fig, ax = plt.subplots()
    sns.histplot(data[column], kde=True, bins=25, ax=ax, color="#90caf9")
    ax.set_title(f'{column} Distribution')
    ax.set_xlabel(column)
    return fig

# -- Confidence Interval --
def get_confidence_interval(data, column, confidence=0.95):
    sample = data[column].sample(50)
    mean = sample.mean()
    sem = stats.sem(sample)
    ci = stats.t.interval(confidence, len(sample)-1, loc=mean, scale=sem)
    return ci

# -- Hypothesis Test --
def hypothesis_test(data, column, group_col):
    groups = data[group_col].unique()
    if len(groups) >= 2:
        g1 = data[data[group_col] == groups[0]][column]
        g2 = data[data[group_col] == groups[1]][column]
        stat, p = stats.ttest_ind(g1, g2)
        return stat, p, groups[0], groups[1]
    return None, None, None, None

# -- Regression (simple & multiple) --
def regression_plot(data, x, y):
    fig, ax = plt.subplots()
    sns.regplot(x=x, y=y, data=data, ax=ax)
    ax.set_title(f'{x} vs {y}')
    return fig

def multiple_regression(data, features, target):
    X = data[features]
    y = data[target]
    model = LinearRegression().fit(X, y)
    return model.coef_, model.intercept_

# -- ANOVA --
def anova_test(data, value_col, group_col):
    groups = [group[value_col].values for name, group in data.groupby(group_col)]
    f_stat, p_val = stats.f_oneway(*groups)
    return f_stat, p_val