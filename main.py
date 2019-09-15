import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import etl
import model
import report


# Compiling login info
DB_TYPE = 'mysql'
DB_DRIVER = 'pymysql'
DB_USER = 'root' # your username in the mysql server
DB_PASS = 'root' # your password in the mysql server
DB_HOST = '0.0.0.0' # change to hostname of your server if on cloud
DB_PORT = '3309' # change accordingly
DB_NAME = 'homestay' # name of your database
POOL_SIZE = 50

SQLALCHEMY_DATABASE_URI = '{0}+{1}://{2}:{3}@{4}:{5}/{6}'.format(DB_TYPE, DB_DRIVER,
                                                              DB_USER,DB_PASS, DB_HOST,
                                                              DB_PORT, DB_NAME)


if __name__ == '__main__':

    # Creating engine with login info
    engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_size=POOL_SIZE, max_overflow=0)
    con = etl.connect_db(engine)
    raw_df = pd.read_sql('''select * from bank.final''', con=con)
    final_df = etl.first_etl(raw_df)
    X, y = etl.second_etl(final_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    classifiers = [XGBClassifier(), RandomForestClassifier()]
    meta = LogisticRegression(solver='lbfgs')
    filename = 'loan_default_stacked.pkl'
    model.stacking_searchcv_save(classifiers, meta, X_train, y_train, filename)

    my_model = pickle.load(open(filename, 'rb'))
    report.score_and_report(my_model, X_test, y_test)