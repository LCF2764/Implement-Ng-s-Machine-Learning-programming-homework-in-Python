import numpy as np
import matplotlib.pyplot as plt


def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    mu_mat = np.tile(mu, (X.shape[0], 1))  # 将mu扩展成mx2的矩阵
    sigma_mat = np.tile(sigma, (X.shape[0], 1))  # 将sigma扩展成mx2的矩阵
    X_norm = (X - mu_mat) / sigma_mat
    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J = 0
    J = np.sum((np.dot(X, theta) - y)**2)/(2*m)
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - alpha/m * np.dot(X.T, np.dot(X, theta) - y)
        J_history[iter] = computeCostMulti(X, y, theta)
    return theta, J_history


def normalEqn(X, y):
    theta = np.zeros((X.shape[0], 1))
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return theta


# ================ Part 1: Feature Normalization ================
print("载入数据...")
data = np.loadtxt('ex1data2.txt', delimiter=",")
X = data[:, 0:2]
y = data[:, 2]
m = len(y)
y = np.reshape(y, (m, 1))

# Print out some data points
print('First 10 examples from the dataset:')
for i in range(10):
    x = X[i:i+1, :]
    y0 = y[i]
    print("x = {},\t y = {}".format(x, y0))
print("="*40)

# Scale features and set them to zero mean
print('Normalizing Features ...')
(X, mu, sigma) = featureNormalize(X)
# add intercept term to X
ones = np.ones((m, 1))
X = np.column_stack((ones, X))  # 把元素为1的列向量并入X中
print("="*40)

# ================ Part 2: Gradient Descent ================
print('Running gradient descent ...')
alpha = 0.01
num_iters = 8500
theta = np.zeros((3, 1))
(theta, J_history) = gradientDescentMulti(X, y, theta, alpha, num_iters)

# plot the convergence graph
plt.plot(range(len(J_history)), J_history, '-')
plt.xlabel('Number of iterations')
plt.ylabel('cost J')
plt.show()
print('Theta computed from gradient descent:')
print(theta)
print("="*40)
# Estimate the price of a 1650 sq-ft, 3 br house
house = np.array([1650, 3])
house_deal = (house-mu)/sigma.reshape(1, 2)
price = np.column_stack((np.ones((1, 1)), house_deal)).dot(theta)[0][0]
print('Predicted price of a 1650 sq-ft, 3 br house \n(using gradient descent):')
print("${}".format(price))
print("="*40)
# ================ Part 3: Normal Equations ================
print('Solving with normal equations...')
data = np.loadtxt('ex1data2.txt', delimiter=",")
X = data[:, 0:2]
y = data[:, 2]
m = len(y)
y = np.reshape(y, (m, 1))
ones = np.ones((m, 1))
X = np.column_stack((ones, X))  # 把元素为1的列向量并入X中
theta = normalEqn(X, y)
print('Theta computed from the normal equations:')
print(theta)
# Estimate the price of a 1650 sq-ft, 3 br house
house = np.array([1, 1650, 3])
price = np.dot(house, theta)[0]
print('Predicted price of a 1650 sq-ft, 3 br house \n(using normal equations):')
print("${}".format(price))
print("="*40)
