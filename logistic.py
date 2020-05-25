# import libraries
import numpy as np


# define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# define cost function
# J(theta) = 1/m (-y^T log(h) - (1-y)^T log(1-h) )
def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)) / y.size


# define z = theta transpose * x
# z = np.dot(X, theta)

# Hypothesis(x) = sigmoid(z)
# h = sigmoid(z)

# predict for test data
# P (Xt | class = 1)
def predict(Xt, theta):
    z = np.dot(np.transpose(Xt), theta)
    return sigmoid(z)


# gradient is the partial derivative of loss function wrt theta
# gradient = np.dot(X.T, (h - y)) / y.size

def grad_desc(X, y, alpha, num_iter):
    # run gradient descent to adjust theta
    #  1. Calculate gradient average
    #  2. Multiply by learning rate alpha
    #  3. Subtract from weights

    # init weights - i choose all to be zero initially for consistent results
    theta = np.zeros(X.shape[1])

    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        theta = theta - alpha * gradient

        # if( i % 50000 == 0):
        #     calc_loss = cost(h, y)
        #     print('iter num: ', i, 'cost : ', calc_loss.mean(), 'theta: ', theta)
    print('iter num: ', i, 'cost : ', cost(h, y).mean(), 'theta: ', theta)
    return theta


def log_reg(X, y, alpha, num_iter, Xt):
    # add intercept
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)
    # add X0=1 to input data
    Xt = np.concatenate(([1], Xt))

    # build a model for each class to decide which one test data belongs to

    # model for class 0
    y_0 = np.copy(y)
    y_0[y == 2] = 1
    y_0 = y_0 - 1
    y_0 = y_0 * -1

    theta_0 = grad_desc(X, y_0, alpha, num_iter)

    # model for class 1
    y_1 = np.copy(y)
    y_1[y == 2] = 0

    theta_1 = grad_desc(X, y_1, alpha, num_iter)

    # model for class 2
    y_2 = np.copy(y)
    y_2[y == 1] = 0
    y_2[y == 2] = 1

    theta_2 = grad_desc(X, y_2, alpha, num_iter)

    # use i th model to decide for c_i

    preds = np.zeros(len(np.unique(y)))
    preds[0] = predict(Xt, theta_0)
    preds[1] = predict(Xt, theta_1)
    preds[2] = predict(Xt, theta_2)

    print("class 0 P = ", preds[0])
    print("class 1 P = ", preds[1])
    print("class 2 P = ", preds[2])

    # choose max
    class_label = np.where(preds == max(preds))

    # print result
    # print("Test point ", Xt, " has label ", class_label[0], " according to logistic regression classification")

    return class_label[0]
