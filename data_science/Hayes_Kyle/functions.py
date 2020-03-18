#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Created on Saturday, March 14 15:23 2020

@author: khayes847
"""
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def kmeans_score(data, n_groups):
    """
    This function will perform KMeans clustering on the included dataset,
    using the n_groups numbers as the number of clusters. It will then
    find and return the predicted clusters' Silhouette Score.

    Parameters:
    data: The dataset in question.
    n_groups (int): The number of clusters.

    Returns:
    score (float): The Silhouette Score for the clustered dataset.
    """
    k_means = KMeans(n_clusters=n_groups, random_state=42)
    k_means.fit(data)
    labels = k_means.labels_
    score = float(silhouette_score(data, labels))
    return score


def agg_score(data, n_groups, score=True):
    """
    Performs Agglomerative Hierarchical Clustering on data using
    the specified number of components. If "Score" is selected,
    returns the Silhouette Score. Otherwise, produces the cluster labels,
    and adds them to the original dataset. For convenience, the function
    also performs the data cleaning steps that don't require the log, outlier-
    capping, or scaling transformations.

    Parameters:
    data: The dataset in question.
    n_groups (int): The number of clusters.
    score (bool): Whether the function will return the Silhouette
                  Score. If 'True', the function will return the Silhouette
                  Score. If 'False', the function will add the clustered labels
                  to the dataset, then save and return the dataset.

    Returns:
    score_val (float): The Silhouette Score for the clustered dataset.
    target: The target labels as a pandas dataframe.
    """
    agg_comp = AgglomerativeClustering(n_clusters=n_groups)
    agg_comp.fit(data)
    labels = agg_comp.labels_
    if score:
        score_val = float(silhouette_score(data, labels))
        return score_val

    data = pd.read_csv("data/shoppers.csv")

    # Combining datasets
    target = pd.DataFrame(labels, columns=['Target'])
    return target


def pca_95(data):
    """
    This function performs PCA dimension reduction on data, in
    order to determine whether doing so will improve clustering.
    The number of dimensions is determined as the number that will
    retain at least 95% variance.

    Parameters:
    data: The dataset in question.

    Returns:
    data: The transformed dataset.
    """
    pca = PCA(n_components=.95, random_state=42)
    pca_array = pca.fit_transform(data)
    data_pca = pd.DataFrame(pca_array)
    return data_pca


def sil_score(data):
    """
    This function performs Agglomerative Hierarchical Clustering and KMeans
    clustering on pre- and post-PCA data into a range of two to ten clusters.
    For each of the four cluster methods, it compiles a list of Silhouette
    Scores at each cluster number, and graphs them using a line graph.
    This is done in order to determine which cluster produces the highest
    Silhouette Score, as well as how many clusters we should use.

    Parameters:
    data: The dataset in question.

    Returns:
    None
    """
    data_pca = pca_95(data)
    n_list = list(range(2, 10))
    kmeans_no_pca = []
    kmeans_pca = []
    agg_no_pca = []
    agg_pca = []
    for number in n_list:
        score = kmeans_score(data, n_groups=number)
        kmeans_no_pca.append(score)
        score = agg_score(data, n_groups=number)
        agg_no_pca.append(score)
        score = kmeans_score(data_pca, n_groups=number)
        kmeans_pca.append(score)
        score = agg_score(data_pca, n_groups=number)
        agg_pca.append(score)
    plot_sil_scores(kmeans_no_pca, agg_no_pca, kmeans_pca, agg_pca, n_list)


def plot_sil_scores(kmeans_no_pca, agg_no_pca, kmeans_pca, agg_pca, n_list):
    """
    Plots Silhouette Scores for KMeans and Agglomerative Hierarchical
    Clustering both pre- and post-PCA against the number of clusters
    used to obtain each score.

    Parameters:
    kmeans_no_pca: The list of Silhouette Scores for
                   the KMeans clustering without PCA.
    agg_no_pca: The list of Silhouette Scores for the
                Agglomerative Hierarchical clustering without PCA.
    kmeans_pca: The list of Silhouette Scores for the
                KMeans clustering with PCA.
    agg_pca: The list of Silhouette Scores for the
             Agglomerative Hierarchical clustering with PCA.
    n_list: A list describing the range of cluster numbers used
           (from two to ten).

    Returns:
    None
    """
    plt.figure(figsize=(16, 8))
    plt.plot(n_list, kmeans_no_pca, label='KMeans')
    plt.plot(n_list, agg_no_pca, label='Agglomerative Hierarchical')
    plt.plot(n_list, kmeans_pca, label='KMeans W/ PCA')
    plt.plot(n_list, agg_pca, label='Agglomerative Hierarchical W/ PCA')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.title("Comparison of Clustering Methods")
    plt.show()


def get_targets(data, n_groups):
    """
    In order to obtain our final set of labels for the data,
    this function transforms the data first using PCA,
    then Agglomerative Hierarchical Clustering. It returns
    the target labels as a pandas dataframe.

    Parameters:
    data: The dataset in question.
    n_groups (int): The number of clusters the function will form.

    Returns:
    target: The target labels as a pandas dataframe.
    """
    data = pca_95(data)
    target = agg_score(data, n_groups, score=False)
    return target


def train_test(x_val, y_val, test=.25, rs_val=42):
    """
    This function separates takes in the feature and target datasets.
    It then splits them into training and test datasets, according
    to the test size and random state specified. It stratifies the
    test datasets according to the target values, in order to
    maintain target value ratios.

    Parameters:
    x_val: The feature dataset.
    y_val: The target dataset.
    test (float): The percentage of the datasets that will be split
                  into the test dataset.
    rs_val (int): The random_state value for the train_test_split
                  function.

    Returns:
    x_train: The features for the training dataset.
    x_test: The features for the test dataset.
    y_train: The targets for the training dataset.
    y_test: The targets for the test dataset.
    """
    x_train, x_test, y_train, y_test = train_test_split(x_val, y_val,
                                                        test_size=test,
                                                        random_state=rs_val,
                                                        stratify=y_val)
    return x_train, x_test, y_train, y_test


def visualize_feature_importance(x_val, y_val):
    """
    In order to determine the important features in the
    classification method, this function creates a random forests
    algorithm and fits it to the included data. It then graphs
    the relative importances of the ten most influential
    features.

    Parameters:
    x_val: The dataset features
    y_val: The dataset labels

    Returns:
    None
    """
    # Determining feature importance using Random Forests
    clf = RandomForestClassifier(n_estimators=100,
                                 random_state=42).fit(x_val, y_val)
    feature_importances = (pd.DataFrame(clf.feature_importances_,
                                        index=x_val.columns,
                                        columns=['importance'])
                           .sort_values('importance', ascending=True))
    feature_importances = feature_importances.iloc[-10:]

    # Graphing feature importance
    ax_val = feature_importances.plot(kind='barh', figsize=(20, 10),
                                      legend=None)
    ax_val.set_xlabel('Importance', fontsize=16)
    ax_val.set_ylabel('Features', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title('Feature Importance Determined By Random Forests', fontsize=20)
    plt.show()


def graph_differences_cat(data, target, features, overlap=False):
    """
    In order to label the target categories properly, we have to describe
    each target group's relationship to the important variables. This feature
    will create three pairs of bar graphs describing the relationship
    (both cumulative and by percentage) between the cluster labels and
    the included categorical variables. If we are plotting 'Browser_Other' and
    'TrafficType_20', for the purposes of determining overlap between these
    features, the function will also describe the relationship between the
    cluster labels and datapoints belonging to both categories using a
    stacked bar graph if 'overlap' is defined as True.

    Parameters:
    data: The dataset features.
    target: The dataset target labels
    features: A list of features to graph.
    overlap (bool): If set to true, will stack the relationship between cluster
                    label and datapoints in both the 'Browser_Other' and
                    'TrafficType_20' categories.

    Returns:
    None
    """
    # pylint: disable=W0612
    # Create "groupby" dataset for graphing
    data = data.join(target)
    if overlap:
        data['Both'] = ((data['Browser_Other'] == 1) &
                        (data['TrafficType_20'] == 1)).astype(int)
        features += ['Both']
    data_grouped = pd.DataFrame(data['Target'].value_counts())
    for col in features:
        data_grouped[col] = pd.DataFrame(data.groupby('Target')[col].sum())
        data_grouped[f'{col}_percentage'] = (data_grouped[col] /
                                             data_grouped['Target'])
    if overlap:
        features = features[:2]

    # Create graphs
    x_pos = [0, 1, 2]
    for col in features:
        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.bar(x_pos, data_grouped[col], color='green')
        if overlap:
            plt.bar(x_pos, data_grouped['Both'], color='blue')
            plt.legend(['Both features', 'Only one feature'])
        plt.xlabel("Cluster label", fontsize=16)
        plt.ylabel("Datapoint Quantity", fontsize=16)
        plt.title("Quantity Per Cluster", fontsize=16)
        plt.xticks(x_pos, data_grouped.index, fontsize=12)
        plt.yticks(fontsize=12)

        plt.subplot(1, 2, 2)
        plt.bar(x_pos, data_grouped[f'{col}_percentage'], color='green')
        if overlap:
            plt.bar(x_pos, data_grouped['Both_percentage'], color='blue')
            plt.legend(['Both features', 'Only one feature'])
        plt.xlabel("Cluster label", fontsize=16)
        plt.ylabel("Datapoint Percentage", fontsize=16)
        plt.title("Percentage Per Cluster", fontsize=16)
        plt.xticks(x_pos, data_grouped.index, fontsize=12)
        plt.yticks(fontsize=12)

        figure.tight_layout(pad=3.0)
        plt.suptitle(col, fontsize=20)
        plt.show()
# pylint: enable=W0612


def cluster_1_composition(data, target):
    """
    For the purposes of fulling understanding the composition of cluster '1',
    this feature will determine the percentage of cluster '1' datapoints that
    belong to both the 'Browser_Other' category and the 'TrafficType_20_Only'
    categories, the percentage that belong to only one category, and the
    percentage that belong to neither category.

    Parameters:
    data: The dataset features.
    target: The dataset target labels.

    Returns:
    None
    """
    data = data.join(target)
    data = data.loc[data['Target'] == 1]
    data['Both'] = ((data['Browser_Other'] == 1) &
                    (data['TrafficType_20'] == 1)).astype(int)
    data['Browser_Other_Only'] = ((data['Browser_Other'] == 1) &
                                  (data['TrafficType_20'] == 0)).astype(int)
    data['TrafficType_20_Only'] = ((data['Browser_Other'] == 0) &
                                   (data['TrafficType_20'] == 1)).astype(int)
    data['Neither'] = ((data['Browser_Other'] == 0) &
                       (data['TrafficType_20'] == 0)).astype(int)
    data = data[['Both', 'Browser_Other_Only',
                 'TrafficType_20_Only', 'Neither']]
    data_grouped = (pd.DataFrame(data.sum(), columns=['Number']))/len(data)

    # Create Graph
    x_pos = [0, 1, 2, 3]
    plt.figure(figsize=(16, 8))
    plt.bar(x_pos, data_grouped['Number'], color='green')
    plt.xlabel("Feature Overlap Category", fontsize=16)
    plt.ylabel("Datapoint Percentage", fontsize=16)
    plt.title('Cluster "1" Distribution', fontsize=16)
    plt.xticks(x_pos, data_grouped.index, fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def plot_continuous(data, target, new_cluster_0=False):
    """
    In order to label the new clusters, we will need to analyze each cluster
    with regards to the most important continuous variables,
    'ProductRelated_Duration', 'ExitRates', and 'ProductRelated'. We will
    divide each into quantiles of 10, and plot the distributions of each
    cluster using bar plots. Since the clusters are unbalanced, we will
    look at the total percentage of each cluster allocated to each quantile.

    Parameters:
    data: The dataset features.
    target: The dataset target labels.
    new_cluster_0 (bool): If True, removes all Cluster "0" values that don't
                          either have 'Browser_Other' or 'TrafficType_20'
                          for easier comparison with Cluster "1".

    Returns:
    None
    """
    data2 = data.join(target)
    features = ['ProductRelated_Duration', 'ExitRates', 'ProductRelated']
    for col in features:
        if new_cluster_0:
            data2 = data2.loc[~((data2['Target'] == 0) &
                                (~((data2['Browser_Other'] > 1) |
                                   (data2['TrafficType_20'] > 1))))]
            data2 = data2.loc[~((data2['Target'] == 2) &
                                (~((data2['Browser_Other'] > 1) |
                                   (data2['TrafficType_20'] > 1))))]
            data2 = data2.reset_index(drop=True)
        data_grouped = pd.DataFrame(data2['Target'])
        data_grouped['quantiles'] = pd.qcut(data2[col],
                                            q=10, labels=list(range(10)))
        enc = OneHotEncoder()
        data_array = enc.fit_transform(data_grouped[['quantiles']]).toarray()
        enc_data = pd.DataFrame(data_array)
        enc_data.columns = [str(x) for x in list(range(10))]
        data_grouped = data_grouped.join(enc_data)
        data_grouped = data_grouped.drop(columns='quantiles')
        data_grouped_new = pd.DataFrame()
        for tar in [0, 1, 2]:
            total = len(data_grouped.loc[data_grouped['Target'] == tar])
            new = data_grouped.loc[data_grouped['Target'] == tar]
            new_grouped = pd.DataFrame(new.groupby('Target').sum())
            new_grouped = new_grouped/total
            data_grouped_new = pd.concat([data_grouped_new, new_grouped])
        (data_grouped_new.T).plot(kind='bar', figsize=(16, 8))
        if new_cluster_0:
            plt.title(f'{col} with Adjusted Cluster "0"')
        else:
            plt.title(col)
        plt.ylabel('Datapoint Percentage')
        plt.xlabel('Quantile')
        plt.show()


def label_clusters(target):
    """
    This function will change the cluster labels
    from numbers to the predetermined text labels.

    Parameters:
    target: The dataset target labels.

    Returns:
    target: The transformed dataset target labels.
    """
    target['Target'] = target['Target'].apply(lambda x:
                                              str("Product-interested "
                                                  "Traffic Type 20 "
                                                  "and/or Rare Browser "
                                                  "Users") if x == 1 else
                                              (str("Browser 8 Users")
                                               if x == 2 else "Others"))
    return target


def log_results(x_train, x_test, y_train, y_test):
    """
    This function will first perform a GridSearchCV to determine
    the logistic regression parameters that will return the optimal
    F1-Micro score. Once it obtains these parameters, it will use
    them to perform a logistic regression, and it will display
    a MatPlotLib confusion matrix.

    Parameters:
    x_train: The training features.
    y_train: The training targets.
    x_test: The test features.
    y_test: The test targets.

    Returns:
    None
    """
    # GridSearchCV
    logreg = LogisticRegression(random_state=42, multi_class='multinomial')
    param_grid = {
        'solver': ['lbfgs', 'sag', 'saga'],
        'C': [1, 2, 3, 4, 5],
        'fit_intercept': [True, False]
    }
    gs_log = GridSearchCV(logreg, param_grid, cv=3, scoring='f1_micro')
    gs_log.fit(x_train, y_train)
    params = gs_log.best_params_

    # Logistic Regression
    logreg = LogisticRegression(random_state=42, multi_class='multinomial',
                                C=params['C'],
                                fit_intercept=params['fit_intercept'],
                                solver=params['solver'])
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)

    # Display Results
    cm_val = confusion_matrix(y_test, y_pred)
    cm_labels = ["Browser_8", "Other", "Traffic/Browser"]
    plot_confusion_matrix(cm_val, cm_labels)
    scores(y_test, y_pred)


def scores(y_test, y_pred):
    """
    Returns accuracy and F1 scores for our algorithm.

    Parameters:
    y_test: Y-test group.
    y_pred: Predictions for Y-test group.

    Returns:
    None
    """
    print('Test Accuracy score: ', accuracy_score(y_test, y_pred))
    print('Test F1 score: ', f1_score(y_test, y_pred, average='micro'))


def plot_confusion_matrix(cm_val, classes,
                          title="Confusion Matrix"):
    """
    This function will take in the confusion matrix values collected
    from the SkLearn confusion matrix program, and will use them to
    create a MatPlotLib confusion matrix.

    Parameters:
    cm_val: Results SkLearn confusion matrix program.
    classes: List of labels for categories.
    title (optional): Confusion matrix title.

    Returns:
    None
    """
    plt.figure(figsize=(16, 8))
    plt.grid(None)
    plt.imshow(cm_val, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='45')
    plt.yticks(tick_marks, classes)
    thresh = cm_val.max()/2
    for i, j in itertools.product(range(cm_val.shape[0]),
                                  range(cm_val.shape[1])):
        plt.text(j, i, cm_val[i, j], horizontalalignment="center",
                 color="white" if cm_val[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True \nlabel', rotation=0)
    plt.xlabel('Predicted label')
