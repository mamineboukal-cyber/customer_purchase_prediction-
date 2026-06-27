from train import knn, nb, lr, X_test_scaled, y_test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
models = {
    "KNN": knn,
    "Naive Bayes": nb,
    "Logistic Regression": lr
}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"\n--- {name} ---")
    print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:   ", round(recall_score(y_test, y_pred), 4))
    print("F1 Score: ", round(f1_score(y_test, y_pred), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))