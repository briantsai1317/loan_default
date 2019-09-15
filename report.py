######################### Scoring and report

from sklearn.metrics import classification_report

def score_and_report(model, x_test, y_test):

    y_pred = model.predict(x_test.values)
    print(classification_report(y_test, y_pred))