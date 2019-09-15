######################### ETL
import numpy as np
import pandas as pd

# connect to mysql database
def connect_db(engine):

    flag = True
    while flag:
        try:
            con = engine.connect()
            flag = False
        except:
            print('unable to connect')
            flag = True
    return con

# Group A and C; B and D together, and convert to 1 and 0
def convert_label_binary(x):
    if x == 'B' or x == 'D':
        return 1
    else:
        return 0

def first_etl(df):

    df = df.rename(index=str, columns={'loan_id': 'Loan_id'})
    # add a column that indicates how many users are using the account
    df['User_count'] = df.groupby('Account_id')['Account_id'].transform('count')

    # add a column that indicates if the account has a credit card
    df['Credit_card'] = df['Card_type'].apply(lambda x: 1 if x else 0)

    # filter out the accounts that do not have a loan
    df_label = df.dropna(subset=['Loan_id'])

    # filter the repeated entries; only keep the owner
    df_final = df_label[df_label['Disp_type'] == 'OWNER']

    # transform columns into datetime format
    date_lst = ['Birth_date', 'Acc_date', 'Loan_date']
    for col in date_lst:
        df_final.loc[:, col] = pd.to_datetime(df_final[:, col])

    df_final.loc[:, 'Age'] = (df_final['Loan_date'] - df_final['Birth_date']).dt.days
    df_final.loc[:, 'Account_age'] = (df_final['Loan_date'] - df_final['Acc_date']).dt.days

    return df_final

def second_etl(df):

    df_binary = df.copy()
    df_binary['Loan_status_binary'] = df_binary['Loan_status'].apply(convert_label_binary)

    cols = ['Min_balance', 'Insurrance_payment',
            'Household_payment', 'Trans_amount', 'Trans_avg_amount', 'Num_credit', 'Num_withdraw',
            'Trans_max_amount', 'Trans_min_amount', 'Avg_balance', 'Enterpreneurs', 'Age', 'Account_age',
            'Avg_salary', 'Unemploy_95', 'Unemploy_96', 'User_count', 'Trans_insurrance',
            'Trans_statement', 'Trans_sanc', 'Trans_household', 'Credit_card',
            'Trans_interest', 'Num_inhabitants', 'Mun_999', 'Mun_1999', 'Mun_9999',
            'Mun_max', 'Num_cities', 'Urban_ratio', 'Crime_95', 'Crime_96',
            'Trans_std_amount',
            'Max_balance', 'Std_balance', 'Loan_status_binary']
    df_binary = df_binary[[*cols]]

    X = df_binary.drop(['Loan_status_binary'], axis=1)
    y = df_binary['Loan_status_binary']

    cor = X.corr().abs()
    upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    X = X.drop(X[to_drop], axis=1)

    return X, y




