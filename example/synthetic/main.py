import numpy as np
from sklearn.model_selection import StratifiedKFold
from getparameters import score
from araf import ARAF
from arafs import ARAF_fs, ARAF_ub
from generatedata import generate_multiclass_ub, generate_multiclass_rdd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from dnn import DNN


def test_synthetic(generate_data, model, criterion, n_freq, n_conf):
    df, dense_features, sparse_features, target, miss_val, task = generate_data()
    X, y = df[sparse_features + dense_features], df[target]

    res_ori, res_fs, res_ub, res_araf = [], [], [], []
    sfolder = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in sfolder.split(X, y):
        # split data into training set and test set
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        # origin
        res_ori.append(score((X_train, X_test, y_train, y_test), model, criterion))
        print(criterion, "of origin", res_ori[-1], len(X_test.columns))

        # ARAF-fs
        araf_fs = ARAF_fs(n_freq, n_conf)
        araf_fs.fit(X_train, y_train)
        X_train_fs = araf_fs.transform(X_train[sparse_features])
        X_test_fs = araf_fs.transform(X_test[sparse_features])
        res_fs.append(score((X_train_fs, X_test_fs, y_train, y_test), model, criterion))
        print(criterion, "of araf-fs", res_fs[-1], len(X_test_fs.columns))

        # ARAF-ub
        araf_ub = ARAF_ub(n_freq, n_conf)
        araf_ub.fit(X_train, y_train)
        X_train_ub = araf_ub.transform(X_train[sparse_features])
        X_test_ub = araf_ub.transform(X_test[sparse_features])
        res_ub.append(score((X_train_ub, X_test_ub, y_train, y_test), model, criterion))
        print(criterion, "of araf-ub", res_ub[-1], len(X_test_ub.columns))

        # ARAF
        araf = ARAF(n_freq, n_conf)
        araf.fit(X_train, y_train)
        X_train_araf = araf.transform(X_train[sparse_features])
        X_test_araf = araf.transform(X_test[sparse_features])
        res_araf.append(score((X_train_araf, X_test_araf, y_train, y_test), model, criterion))
        print(criterion, "of araf", res_araf[-1], len(X_test_araf.columns))

    methods = ["origin", "ARAF-fs", "ARAF-ub", "ARAF"]
    results = [res_ori, res_fs, res_ub, res_araf]
    for method, result in zip(methods, results):
        mean = round(np.mean(result), 4)
        std = round(np.std(result), 4)
        print(criterion, "of", method, "{}+-{}".format(mean, std))
    return res_ori, res_fs, res_ub, res_araf


if __name__ == "__main__":
    # define the models
    svc = SVC(probability=True, gamma="scale", decision_function_shape="ovr", random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    mlp = MLPClassifier(solver="adam", hidden_layer_sizes=(10, 10),
                        random_state=42, max_iter=10000)
    dnn = DNN(epochs=100, learning_rate=0.0005, hidden_layers=[30, 30], verbose=0)
    lr = LogisticRegression(C=1, penalty='l1', solver="liblinear", random_state=42, max_iter=1000)
    load_data = generate_multiclass_rdd
    model = lr  # mlp, dnn, rf, lr
    criterion = "accuracy_score"  # "log_loss", "accuracy_score", "roc_auc_score"
    res_ori, res_fs, res_ub, res_araf = test_synthetic(load_data, model, criterion, 30, 5)
