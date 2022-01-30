import pandas as pd
import numpy as np
from statistics import mode
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


def clean_data():
    for df in df_train, df_test:
        df['Cabin'].fillna('Unknown', inplace=True)
        df['Fare'] = df['Fare'].map(lambda x: np.nan if x == 0 else x)

        # imputing missing/nan values
        classmeans = df.pivot_table('Fare', columns='Pclass', aggfunc='mean')
        df['Fare'] = df[['Fare', 'Pclass']].apply(
            lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)
        meanAge = np.mean(df.Age)
        df.Age = df.Age.fillna(meanAge)
        modeEmbarked = mode(df.Embarked)[0][0]
        df.Embarked = df.Embarked.fillna(modeEmbarked)


def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return np.nan


# replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

# add more features
def newfeat(df):
    # creating a title column from name
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']

    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))

    df['Title'] = df.apply(replace_titles, axis=1)

    # Turning cabin number into Deck
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Deck'] = df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

    # Creating new family_size column
    df['Family_Size'] = df['SibSp'] + df['Parch']

    # Creating Fare per person column
    df['Fare_Per_Person'] = df['Fare'] / (df['Family_Size'] + 1)

    # Age times class
    df['Age*Class'] = df['Age'] * df['Pclass']

    return df


def discretise_numeric(train, test, data_type_dict, no_bins=10):
    N = len(train)
    M = len(test)
    test = test.rename(lambda x: x + N)
    joint_df = train.append(test)
    for column in data_type_dict:
        if data_type_dict[column] == 'numeric':
            joint_df[column] = pd.qcut(joint_df[column], 10, duplicates='drop', labels=False)
            data_type_dict[column] = 'ordinal'
    train = joint_df.ix[range(N)]
    test = joint_df.ix[range(N, N + M)]
    return train, test, data_type_dict


clean_data()

df_train = newfeat(df_train)
df_test = newfeat(df_test)

data_type_dict = {'Pclass': 'ordinal', 'Sex': 'nominal',
                  'Age': 'numeric',
                  'Fare': 'numeric', 'Embarked': 'nominal', 'Title': 'nominal',
                  'Deck': 'nominal', 'Family_Size': 'ordinal',
                  'Fare_Per_Person': 'numeric', 'Age*Class': 'numeric'}

df_train, df_test, data_type_dict = discretise_numeric(df_train, df_test, data_type_dict)

train = df_train.drop(df_train.columns[[8, 9, 10, 13, 15]], axis=1)
test = df_test.drop(df_test.columns[[8, 9, 10, 13, 14, 15]], axis=1)

# separate feature and target variable for train dataset
train_feature = train.drop('Survived', axis=1)
train_target = train['Survived']

# train and validation split
x_train, x_val, y_train, y_val = train_test_split(train_feature, train_target, test_size=0.10, random_state=123)

# CatBoost Model
Cb_model = CatBoostClassifier(iterations=500, learning_rate=0.3, random_seed=99,
                              use_best_model=True, eval_metric='Logloss')

Cb_model.fit(x_train, y_train, cat_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             eval_set=(x_val, y_val), verbose=True)

pred = Cb_model.predict(x_val)
print("accuracy = {}".format(accuracy_score(y_val, pred)))
# Get predicted probabilities for each class
pred_prob = Cb_model.predict_proba(x_val)
print("log_loss = {}".format(log_loss(y_val, pred_prob)))
