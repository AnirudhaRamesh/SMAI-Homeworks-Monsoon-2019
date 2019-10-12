import numpy as np
import matplotlib.pyplot as plt
w1 = np.array([[0, 0], [0, 1], [2, 0], [3, 2], [3, 3], [2, 2], [2, 0]])
w2 = np.array([[7, 7], [8, 6], [9, 7], [8, 10], [7, 10], [8, 9], [7, 11]])

dim1= np.linspace(-5, 15, 100)
dim2= np.linspace(-5, 15, 100)
X_axis, Y_axis = np.meshgrid(dim1, dim2)

boundary = (1.231 * X_axis * X_axis) - (0.6727 * Y_axis * Y_axis)
boundary = boundary + (1.919 * X_axis * Y_axis) - (40.380 * X_axis) - (11.525 * Y_axis) + 204.720

plt.plot(w1[:, 0], w1[:, 1], 'ro')
plt.plot(w2[:, 0], w2[:, 1], 'go')
plt.contour(X_axis,Y_axis,boundary,[0],colors = ('r'), label = 'Original decision boundary')
plt.show()

plt.plot(w1[:, 0], w1[:, 1], 'ro')
plt.plot(w2[:, 0], w2[:, 1], 'go')
plt.contour(X_axis,Y_axis,boundary,[0],colors = ('r'), label = 'Original decision boundary')
boundary = boundary + 2 * np.log(2)
plt.contour(X_axis,Y_axis,boundary,[0],colors = ('g'), label = 'New decision boundary')
plt.title('Red line: Original decision boundary, Green line: New decision boundary')
plt.show()