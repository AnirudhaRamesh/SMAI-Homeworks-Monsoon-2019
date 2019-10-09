import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
mu = 0
sd = 1
dist = np.random.normal(mu, sd, 10000)

def mlefunc(s,k):
    random_cov = []
    length = k
    for sample in range(s):
        random_cov.append(np.random.choice(dist, length))
    class_mean = []
    class_cov = []
    class_data = [[] for _ in range(s)]
    class_eigvals = [] 
    mle_class = []
    for i in range(s):
        class_mean.append(np.mean(random_cov[i],0))
    #     print (random_cov[i])
        covarr = np.cov(random_cov[i], rowvar=False)

    for i in range(s):
        mle = []
        for j in range(k):
            res1 = np.log(1/(np.sqrt(2 * np.pi) * covarr))
            res2 = - (((random_cov[i][j] - class_mean[i]) * (random_cov[i][j] - class_mean[i]))/(2*covarr*covarr) )
            mle.append((res1+res2))
        mle_class.append(max(mle))
#         print (max(mle))
    plt.plot(mle_class,[x for x in range(s)])
	plt.show()

    return np.var(mle_class)

#part 1
_ = mlefunc(10,1000)

#part 2
#change k, keep s constant
s = 100
k = [10,50,100,500,1000,2000,5000]
arr = []
points = []
for i in k:
    arr.append(mlefunc(s,i))
    points.append([i,arr[-1]])
plt.plot(k,arr)
plt.scatter(k,arr,marker='x')
# for i in range(len(k)):
#     temp = (k[i],arr[i])
#     plt.text(k[i],arr[i],str(temp))
temp = (k[0],arr[0])
plt.text(k[0], arr[0], str(temp))
temp = (k[-1],arr[-1])
plt.text(k[-1], arr[-1], str(temp))
plt.show()

#part 3
#change s, keep k constant
k = 100
s = [10,50,100,500,1000,2000,5000]
arr = []
points = []
for i in s:
    arr.append(mlefunc(i,k))
    points.append([i,arr[-1]])
plt.plot(s,arr)
plt.scatter(s,arr,marker='x')
temp = (s[0],arr[0])
plt.text(s[0], arr[0], str(temp))
temp = (s[-1],arr[-1])
plt.text(s[-1], arr[-1], str(temp))

plt.show()