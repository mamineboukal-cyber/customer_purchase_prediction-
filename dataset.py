import pandas as pd
import os
import kagglehub
path = kagglehub.dataset_download("imakash3011/online-shoppers-purchasing-intention-dataset")
df = pd.read_csv(path + '/online_shoppers_intention.csv')

print("shape:",df.shape)
print("head:",df.head())
print("info:",df.info())
print(df['Revenue'].value_counts())


df_encoded = pd.get_dummies(df,columns=['VisitorType','Month'])
print(df_encoded.shape)
print("info:",df_encoded.info())


bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
print(df_encoded.dtypes)


X = df_encoded.drop(columns=['Revenue'])
y = df_encoded['Revenue'].astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("y values:", y.value_counts())