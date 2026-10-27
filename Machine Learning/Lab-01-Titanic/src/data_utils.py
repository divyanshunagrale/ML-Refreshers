import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _is_male(df):
    gender_map = {'male': 1, 'female': 0}
    df['Gender'] = df['Sex'].map(gender_map)


def _extract_title(df):
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
    df['Title'] = df['Title'].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })
    df["Title"] = df["Title"].apply(lambda t: "Other" if t not in common_titles else t)

def _family_size(df):
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

def _is_alone(df):
    df['IsAlone'] = df['FamilySize'] > 1
    df['IsAlone'] = df['IsAlone'].astype(int)

def _fill_missing(df):
    age_median = df.Age.median()
    embarked_mode = df.Embarked.mode()[0]
    df['Age'] = df['Age'].fillna(age_median)
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)


def preprocess(df):
    _is_male(df)
    _extract_title(df)
    _family_size(df)
    _is_alone(df)
    _fill_missing(df)

    input_cols = ['Pclass',	'Age',	'Fare',	'Embarked',	'Gender', 'Title', 'FamilySize', 'IsAlone']
    
    X = df[input_cols]
    y = df['Survived']

    return X, y