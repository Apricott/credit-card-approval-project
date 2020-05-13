import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


def main():
    data = import_data()
    data = clean_data(data)
    data = handle_non_numerical_data(data)
    data = train_test(data)
    # model(data)


def clean_data(data):
    data = data.replace('?', np.nan)
    data['Gender'] = data['Gender'].map({'a': 'male', 'b': 'female'})
    data['Gender'] = data['Gender'].astype('category')
    pd.to_numeric(['Age'], errors='coerce')
    data['Age'] = data['Age'].astype('float64')
    data['Married'] = data['Married'].map({'1': np.nan, 'u': 1, 'y': 0})
    data['Married'] = data['Married'].astype('bool')
    data['BankCustomer'] = data['BankCustomer'].map({'g': 1, 'p': 0, 'gg': np.nan})
    data['BankCustomer'] = data['BankCustomer'].astype('bool')
    data['EducationLevel'] = data['EducationLevel'].astype('category')
    data['Ethnicity'] = data['Ethnicity'].astype('category')
    data['PriorDefault'] = data['PriorDefault'].map({'t': 1, 'f': 0})
    data['PriorDefault'] = data['PriorDefault'].astype('bool')
    data['Employed'] = data['Employed'].map({'t': 1, 'f': 0})
    data['Employed'] = data['Employed'].astype('bool')
    data['DriverLicense'] = data['DriverLicense'].map({'t': 1, 'f': 0})
    data['DriverLicense'] = data['DriverLicense'].astype('bool')
    data['Approved'] = data['Approved'].map({'+': 1, '-': 0})
    data['Approved'] = data['Approved'].astype('bool')

    # bardzo maly odsetek brakow danych, wiec mozemy je usunac
    #print(data.isnull().mean())
    data.dropna(inplace=True)
    #print(data.isnull().sum())

    return data

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df


def import_data():
    data = pd.read_csv('dataset/cc_approvals.data')
    variable_names = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity', 'YearsEmployed',
                      'PriorDefault', 'Employed', 'CreditScore', 'DriverLicense', 'Citizen', 'ZipCode', 'Income', 'Approved']
    data.columns = variable_names
    data.drop(['ZipCode', 'Citizen'], 1, inplace=True)
    return data

def train_test(data):
    y=data.Approved
    x=data.drop('Approved',axis=1)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    #20% danych do testowania, 80% do trenowania

    #print(x_train.head())

    #scaling
    scaler = StandardScaler()
    scaledx_train = scaler.fit_transform(x_train)
    print(scaledx_train)

    return data


if __name__ == '__main__':
    main()