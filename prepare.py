from sklearn.model_selection import train_test_split
from dataset import df_encoded
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

X = df_encoded.drop(columns=['Revenue'])
y = df_encoded['Revenue'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=67,
    stratify=y
)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train values:\n", y_train.value_counts())
print("y_test values:\n", y_test.value_counts())

smote = SMOTE(random_state=67)
X_train_sm , y_train_sm = smote.fit_resample(X_train,y_train)
print("After SMOTE:")
print("X_train shape:", X_train_sm.shape)
print("y_train values:\n", y_train_sm.value_counts())

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_train_scaled min:", X_train_scaled.min())
print("X_train_scaled max:", X_train_scaled.max())

print("X_test_scaled shape:", X_test_scaled.shape)
print("X_test_scaled min:", X_test_scaled.min())
print("X_test_scaled max:", X_test_scaled.max())