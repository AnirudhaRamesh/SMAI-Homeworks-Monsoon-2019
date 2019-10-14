import numpy as np  
import matplotlib.pyplot as plt 
from numpy import linalg as LA

points = np.array([[2,0],[3,2],[6,2],[5,0]])
mean = np.mean(points,axis=0)
p_t = np.transpose(points)
cov = np.cov(p_t)
print(cov)
w,v = LA.eig(cov)

print(w)

print(v)

origin = 4,1

plt.quiver(*origin, v[:,0], v[:,1], color=['r','g'], scale = 6)

plt.plot([2, 5], [0, 0],color = 'black')
plt.plot([2, 3], [0, 2],color = 'black')
plt.plot([3,6], [2,2],color = 'black')
plt.plot([6,5], [2,0],color = 'black')

plt.show()