import matplotlib.pyplot as plt 
import random
import numpy 

# plt.xlim (-10, 10)
# plt.ylim (-10, 10)

x = numpy.linspace (-20, 20, 200)
y = x + random.randint (-5, 5)

random_points = []
random_points.append(numpy.linspace (-19, 19, 1000))

temp = []
for i in random_points[0]:
	# print(i)
	temp.append (i + (random.randint(-3, 3) * random.random()))
random_points.append(temp)

plt.plot (random_points[0], random_points[1], 'go')

plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

covariance_matrix = numpy.cov (random_points[0], random_points[1])

eigen_values, eigen_vectors = numpy.linalg.eig (covariance_matrix)

mean_points = []
mean_points.append(numpy.mean (random_points[0]))
mean_points.append(numpy.mean (random_points[1]))
origin = [mean_points[0]], [mean_points[1]]

plt.quiver (*origin, eigen_vectors[:,0], eigen_vectors[:,1], color=['r', 'b'], scale=2)


plt.show()