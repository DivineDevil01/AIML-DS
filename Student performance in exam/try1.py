from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("StudentsPerformance.csv")

x = df[["parental level of education", "test preparation course","reading score"]]
y = df["math score"]


# x_encoded = pd.get_dummies(x, drop_first=True)
# print(x_encoded.head(10))
# print(x.head(10))

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

x_cat_encode = encoder.fit_transform(
    df[cat_cols]
)

# ecoder_features_name = encoder.get_feature_names_out(cat_cols)

# x_cat_df = pd.DataFrame(
#     x_cat_encode,
#     columns= ecoder_features_name
# )

# print(x_cat_df.head())

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", encoder,cat_cols),
        ("num", StandardScaler(),num_cols)
    ]
)

x_processed = preprocessor.fit_transform(df)

feature_names = (
    list(
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(cat_cols)
    )
    + num_cols
)

# coefficients = pd.Series(model.coef_, index=feature_names)
# print(coefficients.sort_values(ascending=False))

# print(x_processed[:5])

