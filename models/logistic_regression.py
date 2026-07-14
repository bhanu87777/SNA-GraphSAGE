from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_eval_logistic(features, labels, train_mask, test_mask, max_iter=1000, random_state=42):
    X_train = features[train_mask]
    y_train = labels[train_mask]
    X_test = features[test_mask]
    y_test = labels[test_mask]

    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return acc, report
