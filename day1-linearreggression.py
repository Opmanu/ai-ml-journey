import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
# House size in square feet
X = np.array([
    [500],
    [750],
    [1000],
    [1250],
    [1500],
    [1750],
    [2000]
])

# House price in lakhs
y = np.array([20, 30, 40, 50, 60, 70, 80])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Make prediction
size_in_nums=int(input("enter size to predict:"))
size = [[size_in_nums]]
# breakpoint()
print(size)
prediction = model.predict(size)
 
print("House size:", size[0][0], "sqft")
print("Predicted price:", prediction[0], "lakhs")

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
