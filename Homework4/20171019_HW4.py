import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import random, linalg

# case one --------------------------------

mean1a = np.array([1,2,3])
mean1b = np.array([4,5,6])
cov1 = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
xa,ya,za = np.random.multivariate_normal(mean1a,cov1,1000).T
xb,yb,zb = np.random.multivariate_normal(mean1b,cov1,1000).T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.scatter(xa, ya, za, c = 'orange', marker = 'x')
ax.scatter(xb, yb, zb, c = 'green', marker = 'o')
plt.show()
plt.subplot(3,1,1)
plt.title('X-Y Plane')
plt.scatter(xa,ya,c = 'orange', marker = 'x')
plt.scatter(xb,yb,c = 'green', marker = 'o')
plt.subplot(3,1,2)
plt.title('Y-Z Plane')
plt.scatter(ya,za,c = 'orange', marker = 'x')
plt.scatter(yb,zb,c = 'green', marker = 'o')
plt.subplot(3,1,3)
plt.title('Z-X Plane')
plt.scatter(za,xa,c = 'orange', marker = 'x')
plt.scatter(zb,xb,c = 'green', marker = 'o')
plt.show()


# case two --------------------------------

mean2a = np.array([1,3,5])
mean2b = np.array([2,4,6])
A = random.rand(3,3)
B = np.dot(A,A.transpose())
cov2 = B
print(cov2)
xa,ya,za = np.random.multivariate_normal(mean2a,cov2,1000).T
xb,yb,zb = np.random.multivariate_normal(mean2b,cov2,1000).T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.scatter(xa, ya, za, c = 'orange', marker = 'x')
ax.scatter(xb, yb, zb, c = 'green', marker = 'o')
plt.show()
plt.subplot(3,1,1)
plt.title('X-Y Plane')
plt.scatter(xa,ya,c = 'orange', marker = 'x')
plt.scatter(xb,yb,c = 'green', marker = 'o')
plt.subplot(3,1,2)
plt.title('Y-Z Plane')
plt.scatter(ya,za,c = 'orange', marker = 'x')
plt.scatter(yb,zb,c = 'green', marker = 'o')
plt.subplot(3,1,3)
plt.title('Z-X Plane')
plt.scatter(za,xa,c = 'orange', marker = 'x')
plt.scatter(zb,xb,c = 'green', marker = 'o')
plt.show()

# case three --------------------------------

mean3 = np.array([1,2,1])
cov3a = np.matrix([[2,0,0],[0,2,0],[0,0,2]])
cov3b = np.matrix([[2,0,0],[0,1,0],[0,0,3]])
xa,ya,za = np.random.multivariate_normal(mean3,cov3a,1000).T
xb,yb,zb = np.random.multivariate_normal(mean3,cov3a,1000).T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.scatter(xa, ya, za, c = 'orange', marker = 'x')
ax.scatter(xb, yb, zb, c = 'green', marker = 'o')
plt.show()
plt.subplot(3,1,1)
plt.title('X-Y Plane')
plt.scatter(xa,ya,c = 'orange', marker = 'x')
plt.scatter(xb,yb,c = 'green', marker = 'o')
plt.subplot(3,1,2)
plt.title('Y-Z Plane')
plt.scatter(ya,za,c = 'orange', marker = 'x')
plt.scatter(yb,zb,c = 'green', marker = 'o')
plt.subplot(3,1,3)
plt.title('Z-X Plane')
plt.scatter(za,xa,c = 'orange', marker = 'x')
plt.scatter(zb,xb,c = 'green', marker = 'o')
plt.show()

