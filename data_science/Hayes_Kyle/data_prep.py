#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0640
# pylint: disable=W0108
"""
Created on Friday, March 13 14:56 2020

@author: khayes847
"""
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
plt.style.use('ggplot')


def convert_type(data):
    """
    This function converts each categorical feature with a numerical
    data type to a string data type.

    Parameters:
    data: The dataset in question.

    Returns:
    data: The transformed dataset.
    """
#   Categorical features
    columns = ['Browser', 'OperatingSystems', 'Region', 'TrafficType']
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
    return data


def category_grouping(data):
    """
    Each of the features "TrafficType", "OperatingSystems", and "Browser"
    contain categorical values with less than 1% (123) overall datapoints.
    Since these "categorical outliers" could potentially skew a clustering
    algorithm, we will combine each value with ten or fewer datapoints into
    a single "Other" value.

    Parameters:
    data: The dataset in question.

    Returns:
    data: The transformed dataset.
    """
    data['TrafficType'] = data['TrafficType'].apply(lambda x: 'Other' if x in
                                                    ['7', '9', '12', '14',
                                                     '15', '16', '17', '18',
                                                     '19']
                                                    else x)
    data['OperatingSystems'] = data['OperatingSystems'].apply(lambda x: 'Other'
                                                              if x in
                                                              ['4', '5', '6',
                                                               '7', '8']
                                                              else x)
    data['Browser'] = data['Browser'].apply(lambda x: 'Other' if x in
                                            ['3', '7', '9', '11', '12', '13']
                                            else x)
    data['VisitorType'] = data['VisitorType'].apply(lambda x: x if
                                                    x == 'Returning_Visitor'
                                                    else 'New_or_Other')
    return data


def resample_vals(x_train, y_train):
    """
    Prior to running a supervised classification algorithm,
    we will need to even our training target labels through
    resampling. Since many of our values are categorical,
    we will use SMOTENC.

    Parameters:
    x_train: The training dataset features.
    y_train: The training dataset targets.

    Returns:
    x_train_new: The resampled training dataset features.
    y_train_new: The resampled training dataset targets.
    """
#   Specify categorical variables
    cats = [0, 2, 4]
    cats += list(range(10, 18))

#   Resample all non-majority categories
    sm_alg = SMOTENC(categorical_features=cats, random_state=42,
                     sampling_strategy='not majority')
    x_array, y_array = sm_alg.fit_resample(x_train, y_train)
    x_train_new = pd.DataFrame(x_array, columns=list(x_train.columns))
    y_train_new = pd.DataFrame(y_array, columns=list(y_train.columns))
    return x_train_new, y_train_new


def onehot_features(data):
    """
    The dataset contains several nominal categorical features
    with more than two distinct values. Since each
    nominal categorical feature must be binary, we will
    create dummy variables for each feature. In order
    to prevent multicollinearity, we will also remove the
    most common value from each of these features. Note:
    prior to transformation, we convert the Boolean features
    to numerical [0,1] features.

    Parameters:
    data: The dataset in question.

    Returns:
    data: The transformed dataset.
    """

#   Binary Features
    columns = ['Weekend', 'Revenue']
    for col in columns:
        data[col] = data[col].apply(lambda x: float(1) if x else float(0))

    columns = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
               'VisitorType']
    for col in columns:
        enc = OneHotEncoder()
        data_array = enc.fit_transform(data[[col]]).toarray()
        enc_data = pd.DataFrame(data_array)
        enc_data.columns = list(enc.get_feature_names([col]))
        data = data.join(enc_data)

    data = data.drop(columns={'Month', 'Month_May', 'OperatingSystems',
                              'OperatingSystems_2', 'Browser', 'Browser_2',
                              'Region', 'Region_1.0', 'TrafficType',
                              'TrafficType_2', 'VisitorType',
                              'VisitorType_Returning_Visitor'})
    return data


def log_trans(data, test=False):
    """
    The dataset contains several continuous features with strong
    postive skews. In order to increase each feature's normality,
    we will transform each continuous feature using a log function.
    Since each feature has '0' values, we will first determine the
    minumum non-zero value for each feature, divide this value by 2,
    and add the resulting "minimum value" to each zero value.

    This feature has an optional 'x_test' option. If a test dataset
    (x_test) is included, the function will add the determined
    minimum value to each zero value. However, in order to prevent
    leakage, the function will not use the test dataset to determine
    minimum values.

    Parameters:
    data: The dataset in question.
    test (optional, bool): If True, separates test data.

    Returns:
    data: The transformed dataset.
    """
    logs = ['Administrative', 'Administrative_Duration', 'Informational',
            'Informational_Duration', 'ProductRelated',
            'ProductRelated_Duration', 'BounceRates', 'ExitRates',
            'PageValues']
    if test:
        data_test = data.loc[data['Train'] == 0]
        data = data.loc[data['Train'] == 1]

    for col in logs:
        zero_val = float((min(i for i in list(data[col]) if i > 0))/2)
        data[col] = data[col].apply(lambda x: zero_val if x == 0 else x)
        data[col] = data[col].apply(lambda x: np.log(x))
        if test:
            data_test[col] = data_test[col].apply(lambda x:
                                                  zero_val
                                                  if x == 0 else x)
            data_test[col] = data_test[col].apply(lambda x: np.log(x))
    if test:
        data = pd.concat([data, data_test])
    return data


def cap_outliers(data, test=False):
    """
    The dataset contains several continuous features with strong
    outliers. In order to decrease these outliers' influence, we will
    determine for each continuous feature the value for the 99.97%
    quantile. We will then cap each value above this quantile at the
    99.97% level.

    This feature has an optional 'x_test' option. If a test dataset
    (x_test) is included, the function will cap each feature's outliers
    at the determined 99.97% quantile value. However, in order to prevent
    leakage, the function will not use the test dataset to determine
    this value.

    Parameters:
    data: The dataset in question.
    test (optional, bool): If True, separates test data.

    Returns:
    data: The transformed dataset.
    """
    caps = ['Administrative', 'Administrative_Duration', 'Informational',
            'Informational_Duration', 'ProductRelated',
            'ProductRelated_Duration', 'BounceRates', 'ExitRates',
            'PageValues']

    if test:
        data_test = data.loc[data['Train'] == 0]
        data = data.loc[data['Train'] == 1]

    for col in caps:
        outlier_level = float(data[col].quantile(.9997))
        data[col] = data[col].apply(lambda x: outlier_level
                                    if x > outlier_level else x)
        if test:
            data_test[col] = data_test[col].apply(lambda x:
                                                  outlier_level
                                                  if x > outlier_level
                                                  else x)
    if test:
        data = pd.concat([data, data_test])
    return data


def scale(data, test=False):
    """
    The dataset contains multiple features that use different numerical
    ranges, and which have different statistical properties. In order
    to standardize these prior to clustering and/or a regression function,
    we will fit a SciKit StandardScaler algorithm to our data, and then
    create a scaled version of our data.

    This feature has an optional 'x_test' option. If a test dataset
    (x_test) is included, the function will use the fitted StandardScaler
    algorithm to transform the test dataset. To prevent leakage, the test
    dataset will not be used in fitting the StandardScaler algorithm.

    Parameters:
    data: The dataset in question.
    test (optional, bool): If True, separates test data.

    Returns:
    data: The transformed dataset.
    """

    if test:
        data_test = data.loc[data['Train'] == 0]
        data = data.loc[data['Train'] == 1]

    scaler = StandardScaler()
    scaler.fit(data)
    scaled_array = scaler.transform(data)
    data = pd.DataFrame(scaled_array,
                        columns=list(data.columns))
    data['Train'] = 1
    if test:
        test_array = scaler.transform(data_test)
        data_test = pd.DataFrame(test_array, columns=list(data_test.columns))
        data_test['Train'] = 0
        data = pd.concat([data, data_test])
    return data


def clean(data, skip_transformations=False, target=False):
    """
    This function performs each of the appropriate cleaning functions
    on the unsupervised, unlabeled dataset. Since investigating the
    relationship between individual features and new labels will be easier
    without the log transformation, continuous outlier capping, and
    standard scaling, the 'transformations' variable will skip these if
    labelled "True". In addition, if a target variable dataset is included,
    the function will split the dataset into stratified training and test
    datasets, perform oversampling and undersampling on the training set,
    and transform the test features without data leakage.

    Parameters:
    data: The dataset in question.
    skip_transformations (bool): If True, skips log transformation,
                                 continuous outlier capping, and
                                 standard scaling.
    target (boolean, optional): If true, separates target,
                                creates training/testing datasets, and
                                oversamples minority values.

    Returns:
    data: The transformed dataset.
    x_train: The transformed training features.
    y_train: The transformed training targets.
    x_test: The transformed test features.
    y_test: The transformed test targets.
    """
    data = convert_type(data)
    data = category_grouping(data)
    if target:
        target = data[['Target']]
        data = data.drop(columns='Target')
        x_train, x_test, y_train, y_test = f.train_test(data, target)
        x_train, y_train = resample_vals(x_train, y_train)
        x_train = x_train.assign(Train=lambda x: 1)
        x_test = x_test.assign(Train=lambda x: 0)
        data = pd.concat([x_train, x_test])
        data = onehot_features(data)
        data = log_trans(data, test=True)
        data = cap_outliers(data, test=True)
        data = scale(data, test=True)
        x_train = data.loc[data['Train'] == 1]
        x_test = data.loc[data['Train'] == 0]
        return x_train, x_test, y_train, y_test
    data = onehot_features(data)
    if skip_transformations:
        return data
    data = log_trans(data)
    data = cap_outliers(data)
    data = scale(data)
    return data
