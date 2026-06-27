from prepare import X_train_scaled, X_test_scaled, y_train_sm, y_test
from sklearn.neighbors import KNeighborsClassifier
import math

n = len(X_train_scaled)
k = int(math.sqrt(n))
if k % 2 == 0:
    k += 1

print("K value:", k)

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train_sm)

y_pred_knn = knn.predict(X_test_scaled)

print("Done — KNN trained and predicted")
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train_scaled, y_train_sm)

y_pred_nb = nb.predict(X_test_scaled)

print("Done — Naive Bayes trained and predicted")
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=67)
lr.fit(X_train_scaled, y_train_sm)

y_pred_lr = lr.predict(X_test_scaled)

print("Done — Logistic Regression trained and predicted")