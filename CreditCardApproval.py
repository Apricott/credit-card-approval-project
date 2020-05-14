import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def main():
    data = import_data()
    cleaned_data = clean_data(data)
    numerical_data = handle_non_numerical_data(cleaned_data)
    model_data(numerical_data)


def model_data(data):
    y = data.Approved
    x = data.drop('Approved', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    # 20% danych do testowania, 80% do trenowania

    scaledX_train = scale_data(X_train)
    scaledX_test = scale_data(X_test)
    classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, activation='relu',
                               solver='adam', random_state=1)
    classifier.fit(scaledX_train, y_train)
    y_pred = classifier.predict(scaledX_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy of classifier: ", classifier.score(scaledX_test, y_test))


def scale_data(data):
    columns_to_scale = data.loc[:, (data.dtypes == np.float64) | (data.dtypes == np.int64)]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(columns_to_scale)

    return scaled_data


def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if pd.api.types.is_categorical_dtype(df[column].dtype):
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))
            df[column] = df[column].astype('category')

    return df


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

    # print(data.isnull().mean())
    # bardzo maly odsetek brakow danych, wiec mozemy je usunac
    data.dropna(inplace=True)

    return data


def import_data():
    data = pd.read_csv('dataset/cc_approvals.data')
    variable_names = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity',
                      'YearsEmployed',
                      'PriorDefault', 'Employed', 'CreditScore', 'DriverLicense', 'Citizen', 'ZipCode', 'Income',
                      'Approved']
    data.columns = variable_names
    data.drop(['ZipCode', 'Citizen'], 1, inplace=True)
    return data


if __name__ == '__main__':
    main()
