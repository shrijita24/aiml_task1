#import dataset and basic exploration
import pandas as pd
df = pd.read_csv('Titanic-Dataset.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())
#handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
#convert categorical features to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
#normalize/standardize the numerical features
from sklearn.preprocessing import StandardScaler
num_cols = ['Age', 'Fare']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
#visualize with boxplots:
import seaborn as sns
import matplotlib.pyplot as plt
for col in num_cols:
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.show()
#remove outliers using IQR (Interquartile Range)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
df = remove_outliers_iqr(df, 'Age')
df = remove_outliers_iqr(df, 'Fare')

