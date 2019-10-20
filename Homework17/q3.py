import matplotlib.pyplot as plt
import numpy as np

N = 100
a = np.random.random(size = 5) - 1
# a[4] = 0
# a[3] = 0
x = [1,2,3,4,5]
sigma = 100
for i in range(0,N - 5):
  x.append(np.dot(a,np.array(x[i:i+5]) + np.random.normal(0,sigma)))
while max(x) >= 10000:
    a = np.random.random(size = 5) - 1
    x = [1,2,3,4,5]
    sigma = 100
    for i in range(0,N - 5):
      x.append(np.dot(a,np.array(x[i:i+5]) + np.random.normal(0,sigma)))

# print(x)
plt.plot(x[:])
plt.grid(True)
plt.title("Data")
plt.show()
x = np.array(x)
d = 5
def L(w):
  loss = 0
  for k in range(len(x) - 5):
    loss += (x[k] - np.dot(w,x[k:k + d]))**2
  return loss/N

def grad(w):
  gradient = np.zeros(d)
  for k in range(len(x) - 5):
    error = (x[k] - np.dot(w,x[k:k + d]))
    gradient += error * x[k:k+d]
  return gradient

w = np.zeros(d)
max_iter = 10000
eta = 0.000000001
losses = []
for iters in range(max_iter):
  w = w + eta * grad(w)
  losses.append(L(w))
  # print(L(w))
print(a)
print(w)
print(L(w))
y = [1,2,3,4,5]
for i in range(0,N - 5):
  y.append(np.dot(w,np.array(x[i:i+5])))

plt.plot(x,label = "Data")
plt.plot([*y[1:]],label = "Predicted")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(losses)
plt.ylabel("Loss")
plt.grid(True)
plt.show()