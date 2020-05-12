import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = import_data()
    data = clean_data(data)
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
    print(data.isnull().mean())
    data.dropna(inplace=True)
    print(data.isnull().sum())

    return data


def import_data():
    data = pd.read_csv('dataset/cc_approvals.data')
    variable_names = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity', 'YearsEmployed',
                      'PriorDefault', 'Employed', 'CreditScore', 'DriverLicense', 'Citizen', 'ZipCode', 'Income', 'Approved']
    data.columns = variable_names
    data.drop(['ZipCode', 'Citizen'], 1, inplace=True)
    return data


if __name__ == '__main__':
    main()
