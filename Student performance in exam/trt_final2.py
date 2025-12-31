import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("StudentsPerformance.csv")

cat_cols = [
    "parental level of education",
    "test preparation course"
]

num_cols = [
    "reading score"
]

X = df[cat_cols + num_cols]
y = df["math score"]

encoder = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", encoder, cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.5, max_iter=10000)
}

cv_results = {}

for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    cv_results[name] = scores.mean()

    print(f"{name}")
    print(f"  Mean R2 (CV): {scores.mean():.4f}")
    print(f"  Std  R2 (CV): {scores.std():.4f}")
    print("-" * 40)

best_model_name = max(cv_results, key=cv_results.get)
print("Best Model:", best_model_name)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

final_model = models[best_model_name]

final_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", final_model)
    ]
)

final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)

print("Final Test MSE:", mean_squared_error(y_test, y_pred))
print("Final Test R2 :", r2_score(y_test, y_pred))

