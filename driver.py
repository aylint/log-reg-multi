# import libraries
import numpy as np
import pandas as pd
import sklearn.datasets  # to import iris
import matplotlib.pyplot as plt  # %matplotlib inline # this line şs for jupyter notebooks

from logistic import log_reg

# define your variables here!

# define test datapoint
Xt = np.array([6.4, 2.8, 5.6, 2.2])

# define k to use in kNN
k = 5

# logistic regression parameters
num_iter = 250000
alpha = 0.1

# #load iris dataset
iris = sklearn.datasets.load_iris()

# # define working data; features X and labels y
# X = iris.data[:, :]
# y = iris.target

X, y = sklearn.datasets.load_iris(return_X_y=True)

# # Step 1 : Visualize data

colors = ['b', 'r', 'g']
for c in np.unique(y):
    plt.plot(X[y == c, 0], X[y == c, 1], 'o', color=colors[int(c)])
# also print our test datapoint
plt.plot(Xt[0], Xt[1], '*', color="k")
plt.show()  # x0 vs x1

for c in np.unique(y):
    plt.plot(X[y == c, 0], X[y == c, 2], 'o', color=colors[int(c)])
# also print our test datapoint
plt.plot(Xt[0], Xt[2], '*', color="k")
plt.show()  # x0 vs x2

# # more visualization including all 4 features
df = pd.DataFrame(X)
axes = pd.pandas.plotting.scatter_matrix(df, alpha=0.2)
plt.tight_layout()
plt.show()

# call logistic regression and print result
log_reg_result = log_reg(X, y, alpha, num_iter, Xt)
print("Test point ", Xt, " has label ", log_reg_result, " according to logistic regression classification")
print("which is ", iris.target_names[int(log_reg_result)])
