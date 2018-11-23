# Machine Learning Online Class - Exercise 2: Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plotData(X, y):
    index0 = list()
    index1 = list()
    j = 0
    for i in y:
        if i == 0:
            index0.append(j)
        else:
            index1.append(j)
        j = j + 1
    plt.scatter(X[index0, 0], X[index0, 1], marker='o')
    plt.scatter(X[index1, 0], X[index1, 1], marker='+')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'], loc='upper right')
    #plt.show()


def costFunction(initial_theta, X, y, myLambda):
    m = y.shape[0]
    #grad = np.zeros((initial_theta.shape))

    J = np.sum(np.dot((-1*y).T, np.log(sigmoid(np.dot(X, initial_theta))))
               - np.dot((1-y).T, np.log(1 - sigmoid(np.dot(X, initial_theta)))))/m
    #grad = np.dot(X.T, sigmoid(np.dot(X, initial_theta)) - y)/m
    return J  # , grad


def gradient(initial_theta, X, y, myLambda):
    m, n = np.shape(X)
    initial_theta = initial_theta.reshape((n, 1))
    # print(initial_theta.shape)
    #grad = np.zeros((initial_theta.shape))
    grad = np.dot(X.T, sigmoid(np.dot(X, initial_theta)) - y)/m
    #grad = ((X.T).dot(sigmoid(np.dot(X, initial_theta)) - y)) / m
    return grad.flatten()


def plotDecisionBoundary(theta, X, y):
    figure = plotData(X[:, 1:], y)
    m, n = X.shape
    # Only need 2 points to define a line, so choose two endpoints
    if n <= 3:
        point1 = np.min(X[:, 1])
        point2 = np.max(X[:, 1])
        point = np.array([point1, point2])
        plot_y = -1*(theta[0] + theta[1]*point)/theta[2]
        plt.plot(point, plot_y, '-')
        plt.legend(['Admitted', 'Not admitted', 'Boundary'], loc='lower left')
    plt.show()
    return 0


def predict(theta, X):
    m, n = X.shape
    p = np.zeros((m, 1))
    k = np.where(sigmoid(X.dot(theta)) >= 0.5)
    p[k] = 1
    return p

if __name__ == '__main__':
# Load Data
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2].astype(int)
    y = np.reshape(y, (y.shape[0], 1))  # 把y转换成mx1的列向量
# ==================== Part 1: Plotting ====================
    print('Plotting data with + indicating (y = 1) examples \
	       and o indicating (y = 0) examples')
    plotData(X, y)
    print("="*40)
# ============ Part 2: Compute Cost and Gradient ============
# Setup the data matrix appropriately, and add ones for the intercept term
    (m, n) = X.shape
    X = np.column_stack((np.ones((m, 1)), X))
    (m, n) = X.shape
    initial_theta = np.zeros((n, 1))  # nitialize fitting parameters
    myLambda = 0
# Compute and display initial cost and gradient
    cost = costFunction(initial_theta, X, y, myLambda)
    grad = gradient(initial_theta, X, y, myLambda)
    print('Cost at initial theta (zeros):', cost)
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros): ')
    print(grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
    test_theta = np.array([[-24], [0.2], [0.2]])
    cost = costFunction(test_theta, X, y, myLambda)
    grad = gradient(initial_theta, X, y, myLambda)
    print('Cost at initial theta :', cost)
    print('Expected cost (approx): 0.218\n')
    print('Gradient at initial theta : ')
    print(grad)
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')
    print("="*40)

# ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.
    myLambda = 1
# Result = op.minimize(fun=costFunction, x0=initial_theta, args=(X, y, myLambda), method='TNC', jac=gradient)
    Result = op.minimize(fun=costFunction, x0=initial_theta,
                     args=(X, y, myLambda), method='TNC', jac=gradient)
#Result = op.fmin_tnc(func=costFunction,x0=initial_theta,args=(X,y))
# Result = op.fmin_bfgs(f=costFunction,x0=initial_theta,fprime=gradient(X,y，1),gtol=1e-5,\disp=1)
    theta = Result.x
    cost = Result.fun
    print('Cost at theta found by fminunc:', cost)
    print('Expected cost (approx):0.203\n')
    print('theta:', theta)
    print('Expected theta (approx):[-25.161  0.206  0.201]\n')
    plotDecisionBoundary(theta, X, y)
    print('='*40)
#  ============== Part 4: Predict and Accuracies ==============
    sample = np.array([1, 45, 85])
    prob = sigmoid(np.dot(sample, theta))
    print('For a student with scores 45 and 85, we predict an admission\
	       probability of ', prob)
    print('Expected value: 0.775 +/- 0.002\n\n')
# compute accuracy on our training set
    p = predict(theta, X)
    accuracy = np.mean(np.double(p == y)) * 100
    print('Train Accuracy:', accuracy)
    print('Expected accuracy (approx): 89.0')
    print('='*40)
