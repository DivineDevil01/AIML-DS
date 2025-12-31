from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


df = pd.read_csv("StudentsPerformance.csv")

x = df[["reading score", "test preparation course", "parental level of education"]]
y = df["math score"]

x_encoded = pd.get_dummies(x, drop_first=True)
print(x.columns)
print(x_encoded.columns)

x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Mean squared error is : ", mse)
print("r2 score is :",r2)

cofficient = pd.Series(model.coef_, index = x_encoded.columns)
print(cofficient.sort_values(ascending=False))