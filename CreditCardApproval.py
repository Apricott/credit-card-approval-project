import pandas as pd
import numpy as np


def main():
    data = import_data()
    data = clean_data(data)
    # model()


def clean_data(data):
    print(data.info())
    data['Gender'] = data['Gender'].map({'a': 'male', 'b': 'female', '?': np.nan})
    data['Gender'] = data['Gender'].astype('category')
    print(data.info())
    # print(data['Male'].head())
    return data


def import_data():
    data = pd.read_csv('dataset/cc_approvals.data')
    print(data.head())
    variable_names = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity', 'YearsEmployed',
                      'PriorDefault', 'Employed', 'CreditScore', 'DriverLicense', 'Citizen', 'ZipCode', 'Income', 'Approved']
    data.columns = variable_names
    return data


if __name__ == '__main__':
    main()
