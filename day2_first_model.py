from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Features
X = [
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
]

# Target
y = [
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)
print("Actual:", y_test)
print("Accuracy:", accuracy)

# Predict a new value
new_student = [[100]]
prediction = model.predict(new_student)

print("Student studied 6 hours:", prediction)