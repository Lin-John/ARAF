import math
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from process import preprocess_X, preprocess_y, lbe_transform, split_transform, araf_transform
from getparameters import score
from loaddata import load_data_lg, load_data_bg
from getparameters import grid_search
from dnn import DNN


def test_bf(load_data, model, criterion):
    X_train, X_test, y_train, y_test, dense_features, sparse_features, target, miss_val, task = load_data()
    X_train, X_test = preprocess_X(X_train, X_test, dense_features)
    y_train, y_test = preprocess_y(y_train.values.ravel(), y_test.values.ravel())

    # **************** select parameters *******************
    X_train1, X_valid = X_train.iloc[:599], X_train.iloc[600:]
    y_train1, y_valid = y_train[:599], y_train[600:]

    n_splits = range(2, 50)
    n_candidate = 10
    k = int(math.sqrt(len(sparse_features) + len(dense_features)))
    n_confs = [i for i in range(50)]
    n_freqs = [150 for i in n_confs]
    scores = grid_search(X_train1, X_valid, y_train1, y_valid, dense_features, sparse_features, miss_val,
                         n_splits, n_candidate, n_freqs, n_confs, model, criterion)
    n_split_best, n_conf_best, score_best = 0, 0, float('inf') if criterion == "log_loss" else -float('inf')
    for n_split, scores_split in zip(n_splits, scores):
        for n_conf, score_conf in zip(n_confs, scores_split):
            if ((criterion == "log_loss" and (
                    score_conf < score_best or (score_conf == score_best and n_conf > n_conf_best))) or
                    (criterion != "log_loss" and (
                            score_conf > score_best or (score_conf == score_best and n_conf > n_conf_best)))):
                n_split_best, n_conf_best, score_best = n_split, n_conf, score_conf
    print(n_split_best, n_conf_best)
    # **************** end select parameters *******************

    # discretize continuous features
    X_train_de, X_test_de, sparse_features_de = split_transform(X_train, X_test, y_train,
                                                                dense_features, sparse_features,
                                                                n_split_best, n_candidate)

    # add associations rules
    X_train_araf, X_test_araf = araf_transform(X_train_de, X_test_de, y_train, sparse_features_de, miss_val,
                                               3 * n_conf_best, n_conf_best)

    # araf
    res_araf = score((X_train_araf, X_test_araf, y_train, y_test), model, criterion)
    print(criterion, "of araf", res_araf)

    # original
    X_train_org, X_test_org = lbe_transform(X_train, X_test, sparse_features)
    res_org = score((X_train_org, X_test_org, y_train, y_test), model, criterion)
    print(criterion, "of origin", res_org)
    return res_araf, res_org, scores


if __name__ == "__main__":
    # define the models
    svc = SVC(probability=True, gamma="scale", decision_function_shape="ovr", random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    mlp = MLPClassifier(solver="adam", hidden_layer_sizes=(10, 10),
                        random_state=42, max_iter=10000)
    dnn = DNN(epochs=100, learning_rate=0.0005, hidden_layers=[30, 30], verbose=0)
    lr = LogisticRegression(C=1, penalty='l1', solver="liblinear", random_state=42, max_iter=1000)

    load_data = load_data_bg
    model = lr  # mlp, dnn, rf, lr
    criterion = "accuracy_score"  # "log_loss", "accuracy_score", "roc_auc_score"
    res_org, res_araf, scores = test_bf(load_data, model, criterion)
