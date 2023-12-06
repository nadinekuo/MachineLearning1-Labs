import numpy as np
import prtools as pr
import matplotlib.pyplot as plt

# Generate input in 2D (1000 x 2) and output (1000 x 1), where x is from std normal
train_size = 1000
x = np.random.randn(train_size, 2)
# y = 50 * np.sin(x[:, 0].reshape(-1, 1)) * np.sin(x[:, 1].reshape(-1, 1)) + np.random.randn(train_size, 1)
# NOTE: 
y = x[:, 0].reshape(-1, 1) + x[:, 1].reshape(-1, 1)

# Split dataset into training and test set 50/50
[train_x, test_x] = np.split(x, 2)
[train_y, test_y] = np.split(y, 2)
train_data = pr.gendatr(train_x, train_y)  # Training regression dataset with x and y values
pr.scatterr(train_data)

# TODO: Use linearr to fit a linear model to the data and have a look at the data from diff points of view by running .py in terminal
R = pr.linearr(train_data, 3)
print(R)
pr.plotr(R, color='r', gridsize=50)

# TODO: Measure error on separate test set for different polynoamial degrees -> error does not change much!
test_data = pr.gendatr(test_x, test_y)
err = pr.testr(test_data, R, 'mse')
print('Error: ', err)

plt.show()