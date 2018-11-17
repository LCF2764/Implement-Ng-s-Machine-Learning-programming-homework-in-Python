from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# ============= warmUpExercise ====================


def warmUpExercise(D):  # Part 1
    print(np.eye(D))


def Plotting_data(X, y):  # Part 2
    plt.scatter(X, y, marker='x', c='r')
    # plt.show()


def computeCost(X, y, theta):
    m = len(y)
    # y = np.reshape(y, (m, 1))  # 原来的y是(m,)的，要把它reshap为(m,1)，否则会得到错误的结果
    J = 0
    J = np.sum((X.dot(theta) - y)**2)/(2*m)
    return J


def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.ones((iterations, 1))

    for iter in range(iterations):
        h = np.dot(X, theta)
        theta = theta - alpha/m * np.dot((X.T), h-y)
        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history


# ============= Part 1 : Basic Function ============ #
# Complete warmUpExercise
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix:\n')
warmUpExercise(5)
print('\n Program paused. Press enter to continue.\n ')

# ============= Part 2 : Plotting  ================= #
print('Plotting Data ... \n')
data = np.loadtxt('ex1data1.txt', delimiter=',')  # load data
X = data[:, 0]
y = data[:, 1]  # 注意这里得到的y的shape为(m, ),要把它reshape为(m,1)
y = np.reshape(y, (len(y), 1))
Plotting_data(X, y)

# ========Part 3: Cost and Gradient descent========= #
ones = np.ones(len(X))  # 生成全部元素为1的列向量
X = np.column_stack((ones, X))  # Add a column of ones to X
theta = np.zeros((2, 1))  # initialize fitting parameters
# Some gradient descent settings
iterations = 1500
alpha = 0.01
print('\nTesting the cost function ...\n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed =', J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, np.array([[-1], [2]]))
print('\nWith theta = [-1 ; 2]\nCost computed =', J)
print('Expected cost value (approx) 54.24\n')
# run gradient descent
(theta, J_history) = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# plot the line fit

plt.plot(X[:, 1], np.dot(X, theta), '-')
plt.show()

# predict values for population sizes of 35000 and 70000
predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of ', predict1*10000)
predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of ', predict2*10000)

# ====== Part 4:Visualizing J(theta_0, theta_1)=====
print('\nVisualizing J(theta_0, theta_1) ...\n')
theta_0_vals = np.linspace(-10, 10, 100)
theta_1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta_0_vals), len(theta_1_vals)))
# Fill out J_vals
for i in range(len(theta_0_vals)):
    for j in range(len(theta_1_vals)):
        theta_t = np.column_stack((theta_0_vals[i], theta_1_vals[j]))
        J_vals[i, j] = computeCost(X, y, theta_t.T)

J_vals = J_vals.T

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(theta_0_vals, theta_1_vals, J_vals,
                rstride=1, cstride=1, cmap='rainbow')

plt.xlabel('theta_0')
plt.ylabel('theta_1')

plt.figure()
plt.contourf(theta_0_vals, theta_1_vals, J_vals,
             20, alpha=0.6, cmap=plt.cm.hot)
plt.plot(theta[0], theta[1], c='r', marker='x', markerSize=10, LineWidth=2)
plt.show()
