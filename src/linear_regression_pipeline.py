"""
Linear Regression: House Price Prediction (RealEstateCo)

This script:
1) Creates the dataset
2) Splits into train/test sets
3) Trains a Linear Regression model
4) Evaluates performance using Mean Squared Error (MSE) via cross-validation
5) Plots actual values and the fitted regression line
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression


def build_dataset() -> pd.DataFrame:
    """Create the house size / house price dataset from the scenario."""
    df = pd.DataFrame(
        {
            "houseSize_sqm": [50, 80, 100, 120, 150, 200, 220, 250, 300, 350],
            "housePrice_dllrs": [150, 200, 240, 270, 300, 360, 400, 430, 500, 550],
        }
    )
    return df


def main() -> None:
    # 1) Data Creation
    df = build_dataset()
    print("Dataset preview:")
    print(df.head(), "\n")

    # Features (X) and Target (y)
    X = df[["houseSize_sqm"]]
    y = df["housePrice_dllrs"]

    # 2) Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1
    )
    print("Train size:", X_train.shape, y_train.shape)
    print("Test size:", X_test.shape, y_test.shape, "\n")

    # 3) Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Quick sanity-check prediction on the first test record
    first_pred = model.predict(X_test.iloc[[0]])[0]
    first_actual = y_test.iloc[0]
    print(f"First test prediction: {first_pred:.2f}")
    print(f"First test actual:     {first_actual:.2f}\n")

    # 4) Evaluate Model Performance with MSE (Cross-Validation)
    mse_scores_neg = cross_val_score(
        LinearRegression(),
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=5
    )

    avg_mse = abs(mse_scores_neg.mean())
    print("Negative MSE scores:", mse_scores_neg)
    print("Average Negative MSE:", mse_scores_neg.mean())
    print("Average MSE:", avg_mse, "\n")

    # 5) Plot Actual vs Predicted (Fitted Line)
    y_pred_full = model.predict(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(X["houseSize_sqm"], y, label="Actual Values")
    plt.plot(X["houseSize_sqm"], y_pred_full, label="Fitted Line")

    plt.xlabel("House Size (sqm)")
    plt.ylabel("House Price ($1000)")
    plt.title("Linear Regression: House Size vs House Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
