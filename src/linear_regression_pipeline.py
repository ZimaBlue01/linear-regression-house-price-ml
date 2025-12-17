import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 1. Data Creation
house_size = np.array([50, 80, 100, 120, 150, 200, 220, 250, 300, 350]).reshape(-1, 1)
house_price = np.array([150, 200, 240, 270, 300, 360, 400, 430, 500, 550])


# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    house_size, house_price, test_size=0.2, random_state=42
)


# 3. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)


# 4. Predictions
y_pred = model.predict(X_test)


# 5. Evaluate Model Performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")


# 6. Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.scatter(X_test, y_pred, color="red", label="Predicted Prices")
plt.xlabel("House Size (sqm)")
plt.ylabel("House Price ($1000)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.show()
