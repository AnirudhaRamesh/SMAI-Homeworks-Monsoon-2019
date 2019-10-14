import matplotlib.pyplot as plt
import numpy as np


def dell_huber(w,delta,x,y):

	total = np.array([0.0,0.0])
	for i in range(0,1000):
		if abs(y[i] - np.dot(w,np.array([x[i],1]))) < delta:
			total += -(y[i] - np.dot(w,np.array([x[i],1])) * np.array([x[i],1]))
		else:
			total += -delta * np.sign(y[i] - np.dot(w,np.array([x[i],1]))) * np.array([x[i],1])

	return total/1000



def dell_logcosh(w,delta,x,y):

	total = np.array([0.0,0.0])
	for i in range(0,1000):
		total += np.tanh(np.dot(w,np.array([x[i],1])) - y[i]) *(np.array([x[i],1]))

	return total/1000

def dell_l1(w,delta,x,y):

	total = np.array([0.0,0.0])
	for i in range(0,1000):
		total += np.sign(np.dot(w,np.array([x[i],1])) - y[i]) * np.array([x[i],1])
	return total/1000


def loss_huber(w,delta,x,y):
	total = np.array([0.0,0.0])
	for i in range(0,1000):
		if abs(y[i] - np.dot(w,np.array([x[i],1]))) < delta:
			total += 0.5 * (y[i] - np.dot(w,np.array([x[i],1])))*(y[i] - np.dot(w,np.array([x[i],1])))
		else:
			total += delta * np.abs(y[i] - np.dot(w,np.array([x[i],1]))) - 0.5 * delta * delta

	return total/1000

def loss_logcosh(w,delta,x,y):
	total = np.array([0.0,0.0])
	for i in range(0,1000):
		total += np.log(np.cosh(y[i] - np.dot(w,np.array([x[i],1]))))

	return total/1000	

def loss_l1(w,delta,x,y):

	total = np.array([0.0,0.0])
	for i in range(0,1000):
		total += np.abs(y[i] - np.dot(w,np.array([x[i],1])))

	return total/1000




x = np.arange(-10,10,0.02)
y = np.arange(-10,10,0.02)

y = y+np.random.normal(0,1,1000)



delta = 10
# w = np.array([10,10])
w = 10 * np.random.rand(1,2)[0] - np.array([5,5])


loss = []
w_x = []
w_y = []

alpha = 0.01
for i in range(0,1000):
	### Change here #####
	w = w - alpha*dell_l1(w,delta,x,y)

	w_x.append(w[0])
	w_y.append(w[1])
	#### Change here #####
	loss.append(loss_l1(w,delta,x,y))


print(w)



plt.plot(x,y,"b.")
plt.plot([-10,10],[np.dot(w,np.array([-10,1])),np.dot(w,np.array([10,1]))])
# plt.title('Data Huber')
# plt.title('Data Logcosh')
plt.title('Data L1')
plt.show()

plt.plot(loss)
plt.grid(True)

##### Select One #####
# plt.title('Loss using Huber')
# plt.title('Loss using Logcosh')
plt.title('Loss using L1')
plt.show()

plt.plot(w_x,w_y)
plt.grid(True)

##### Select One #####
# plt.title('Gradient Descent using Huber')
# plt.title('Gradient Descent using Logcosh')
plt.title('Gradient Descent using L1')
plt.show()