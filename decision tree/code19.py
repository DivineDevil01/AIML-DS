# Decision Tree on a Real Dataset (Breast Cancer - Classification)


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Load real dataset
data = load_breast_cancer()
X = data.data           # features
y = data.target         # label (0/1)
feature_names = data.feature_names
target_names = data.target_names  # typically: ['malignant', 'benign']

print("Dataset shape:", X.shape)
print("Classes:", target_names)

# 2) Split into train and test (supervised learning)
# Training data: model learns
# Test data: model is evaluated on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Build Decision Tree model
# max_depth controls overfitting 
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# 4) Train the model
model.fit(X_train, y_train)

# 5) Predict on test data
y_pred = model.predict(X_test)

# 6) Evaluate
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=target_names
))

# 7) Check which features mattered most (important for explainability)
importances = model.feature_importances_
top = sorted(zip(importances, feature_names), reverse=True)[:10]

print("\nTop 10 Important Features:")
for imp, name in top:
    print(f"{name}: {imp:.4f}")

# 8) Predict for a new sample (inference)
# We'll take one row from test set as an example
sample = X_test[0].reshape(1, -1)
pred_class = model.predict(sample)[0]
pred_prob = model.predict_proba(sample)[0]

print("\nSample prediction:")
print("Predicted class:", target_names[pred_class])
print("Probabilities [malignant, benign]:", pred_prob)
