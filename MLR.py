import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the data
# if this doesnt work, try the full path of where your data is
data = pd.read_csv(
  r"Real estate.csv"
)

X = data.iloc[:, 1:-1].to_numpy(dtype=float)
y = data.iloc[:, -1].to_numpy(dtype=float)

# Z-score
X = (X - X.mean(axis=0))/X.std(axis=0)

# loss function
def loss(m, b, X, y):
  return((y - (X@m + b))**2).mean()

# gradient descent
def gradient_descent(m, b, X, y, rate):
  n = len(y)

  error = y - (X@m +b)

  m_gradeint = -(2/n) * ((X.T)@error)
  b_gradient = -(2/n) * error.sum()

  m = m - rate*m_gradeint
  b = b - rate*b_gradient

  return m, b

# values
m = np.zeros(np.shape(X[1, :]))
b = 0.0

epsilon = 1e-8
rate = 0.0001
epochs = 100000000 # not really important what the value is since we have relative improvement

prev_loss = loss(m, b, X, y)
for i in range(epochs):
  m, b = gradient_descent(m, b, X, y, rate)
  current_loss = loss(m, b, X, y)
  relative_improvement = (prev_loss - current_loss)/max(prev_loss, 1e-12)
  if relative_improvement < epsilon:
    print(f"Reached Loss Relative Minimum at {i} Itterations"
          f"\nFinal Loss: {current_loss}"
          f"\nWeights: {m}"
          f"\nBias: {b}"
          )
    break
  if i%100 == 0:
    print(f"Epochs: {i}, Loss: {current_loss:.4f}")
  prev_loss = current_loss


# plotting
pred = X@m + b

limits = [
    min(y.min(), pred.min()),
    max(y.max(), pred.max())
]

plt.scatter(y, pred, alpha=0.6, color="black")
plt.plot(limits, limits, color="red", label="Perfect predictions")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Linear Regression: Actual vs Predicted Prices")
plt.tight_layout()
plt.legend()
plt.show()