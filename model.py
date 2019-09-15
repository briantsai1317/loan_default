######################### Modeling

# Import packages
import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.classifier import StackingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# Stacking models
def stacking_searchcv_save(classifiers, meta, x_train, y_train, filename):

    sclf = StackingClassifier(classifiers=classifiers, meta_classifier=meta)
    pipeline = Pipeline([('sampling', SMOTE(random_state=42)),
                         ('clf', sclf)])
    params = {'clf__xgbclassifier__eta': np.arange(0.11, 0.12, 0.01),
              'clf__xgbclassifier__max_depth': np.arange(11, 12),
              'clf__xgbclassifier__min_child_weight': np.arange(4, 6),
              'clf__xgbclassifier__gamma': np.arange(3, 6),
              'clf__xgbclassifier__alpha': np.arange(3, 6),
              'clf__randomforestclassifier__n_estimators': np.arange(800, 1000, 100),
              'clf__randomforestclassifier__criterion': ['gini', 'entropy'],
              'clf__randomforestclassifier__max_features': ['sqrt', 'log2'],
              'clf__meta_classifier__C': np.arange(0.4, 0.6, 0.1)
              }
    stacked_grid = RandomizedSearchCV(pipeline, param_distributions=params,
                                      cv=5, refit=True, verbose=0, n_jobs=-1)
    stacked_grid.fit(x_train.values, y_train.values)
    pickle.dump(stacked_grid.best_estimator_, open(filename, 'wb'))