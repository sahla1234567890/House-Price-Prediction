
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Step 1: Load the dataset
data = pd.read_csv("house_data.csv")

print("Dataset loaded successfully!")
print(data.head())

# Step 2: Select input and output
X = data[["Area", "Bedrooms", "Bathrooms"]]
y = data["Price"]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create the ML model
model = LinearRegression()

# Step 5: Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# Step 6: Test the model
predictions = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Step 8: Save the trained model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
