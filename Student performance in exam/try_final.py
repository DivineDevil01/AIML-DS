import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
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

encoder = OneHotEncoder(
   categories=[
        [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ],
    [
        "completed",
        "none"
    ]
   ], 
   drop="first",
   sparse_output= False
)


preprocessor = ColumnTransformer(
    transformers= [
        ("cat", encoder, cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

model_piplined = Pipeline(
    steps= [
        ("preprocessing", preprocessor),
        ("model", LinearRegression())
    ]
)

x = df[cat_cols + num_cols]
y = df["math score"]

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.25,random_state=42
)

model_piplined.fit(x_train,y_train)

y_pred = model_piplined.predict(x_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

linear_model = model_piplined.named_steps["model"]
preprocessor = model_piplined.named_steps["preprocessing"]

cat_features_names = (
    preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)

)

all_feature_names = list(cat_features_names) + num_cols

coefficients = pd.Series(
    linear_model.coef_,
    index=all_feature_names
)

print(coefficients.sort_values(ascending=False))
