import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from process import preprocess, lbe_transform, ohe_transform, split_transform
from getparameters import score, get_n_class
from araf import ARAF
from loaddata import load_data_adult, load_data_hd, load_data_dccc, load_data_ce
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from dnn import DNN


def test_uci(load_data, model, criterion):
    # load data
    df, dense_features, sparse_features, target, miss_val, task = load_data()
    X, y = df[sparse_features + dense_features], df[target]
    k = int(math.sqrt(len(sparse_features)+len(dense_features)))

    res_lbe, res_ohe, res_araf_s, res_araf_l = [], [], [], []
    sfolder = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in sfolder.split(X, y):
        # split data into training set and test set
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test, dense_features, target)

        # split training set into new training set and valid set
        X_train1, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42,
                                                                stratify=y_train)

        # get the number of classes after discretizing
        n_splits = range(2, 10)
        n_candidate = 10
        n_conf = 3 * k
        n_freq =  k
        n_split_best = get_n_class(X_train1, X_valid, y_train1, y_valid, dense_features, sparse_features, miss_val,
                                   n_splits, n_candidate, n_freq, n_conf, model, criterion)

        # discretize continuous features
        X_train_de, X_test_de, sparse_features_de = split_transform(X_train, X_test, y_train,
                                                                    dense_features, sparse_features,
                                                                    n_split_best, n_candidate)

        araf = ARAF(30 * k, 10 * k)
        araf.fit(X_train_de[sparse_features_de], y_train, miss_val)

        # LabelEncoded original data
        X_train_lbe, X_test_lbe = lbe_transform(X_train, X_test, sparse_features)
        res_lbe.append(score((X_train_lbe, X_test_lbe, y_train, y_test), model, criterion))
        print(criterion, "of LabelEncoded origin", res_lbe[-1], len(X_test_lbe.columns))

        # OneHotEncoded encoded original data
        X_train_ohe, X_test_ohe = ohe_transform(X_train, X_test, sparse_features)
        res_ohe.append(score((X_train_ohe, X_test_ohe, y_train, y_test), model, criterion))
        print(criterion, "of OneHotEncoded origin", res_ohe[-1], len(X_test_ohe.columns))

        # ARAF-L
        X_train_araf_s = pd.concat((X_train_lbe, araf.transform(X_train_de[sparse_features_de])), axis=1)
        X_test_araf_s = pd.concat((X_test_lbe, araf.transform(X_test_de[sparse_features_de])), axis=1)
        res_araf_s.append(score((X_train_araf_s, X_test_araf_s, y_train, y_test), model, criterion))
        print(criterion, "of araf-l", res_araf_s[-1], len(X_test_araf_s.columns))

        # ARAF-O
        X_train_araf_l = pd.concat((X_train_ohe, araf.transform_inter(X_train_de[sparse_features_de])), axis=1)
        X_test_araf_l = pd.concat((X_test_ohe, araf.transform_inter(X_test_de[sparse_features_de])), axis=1)
        res_araf_l.append(score((X_train_araf_l, X_test_araf_l, y_train, y_test), model, criterion))
        print(criterion, "of araf-o", res_araf_l[-1], len(X_test_araf_l.columns))

    methods = ["LabelEncoded", "OneHotEncoded", "ARAF-L", "ARAF-O"]
    results = [res_lbe, res_ohe, res_araf_s, res_araf_l]
    for method, result in zip(methods, results):
        mean = round(np.mean(result), 4)
        std = round(np.std(result), 4)
        print(criterion, "of", method, "{}+-{}".format(mean, std))
    return res_lbe, res_ohe, res_araf_s, res_araf_l


if __name__ == "__main__":
    # define the models
    svc = SVC(probability=True, gamma="scale", decision_function_shape="ovr", random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    mlp = MLPClassifier(solver="adam", hidden_layer_sizes=(10, 10),
                        random_state=42, max_iter=10000)
    dnn = DNN(epochs=100, learning_rate=0.0005, hidden_layers=[30, 30], verbose=0)
    lr = LogisticRegression(C=1, penalty='l1', solver="liblinear", random_state=42, max_iter=1000)

    load_data = load_data_hd
    model = lr  # mlp, dnn, rf, lr
    criterion = "log_loss"  # "log_loss", "accuracy_score", "roc_auc_score"
    res_lbe, res_ohe, res_araf_s, res_araf_l = test_uci(load_data, model, criterion)
