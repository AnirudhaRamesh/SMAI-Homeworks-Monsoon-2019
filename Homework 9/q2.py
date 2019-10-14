import matplotlib.pyplot as plt 
import random 

def f(x):
    return x**2

def df(x):
    return 2*x

def gradientDescent(initial_value,learning_rates):
    plt.figure()
    final_x = [] 
    first_5 = [] 
    iter_num = [] 
    for i in learning_rates:
        x = initial_value
        iterations = 0 
        descent_list = [] 
        prev_x = 0
        temp = abs(x - prev_x)
        while temp < 10 and temp > 0.001:
            descent_list.append(f(x))
            iterations += 1
            if iterations <= 6:
                first_5.append(f(x))
            if iterations >= 100:
                break
            prev_x = x
            x = x - i*df(x)
            temp = abs(x - prev_x)

        iter_num.append(iterations)
        final_x.append([i,x])
        plt.plot(descent_list, label = i)
    plt.legend(loc="upper right")
    plt.show()
        
    return final_x, first_5, iter_num

nu = [0.1]
final_x, first_5,_ = gradientDescent(-2, nu)

# plt.scatter([x for x in range(6)], first_5)
plt.show()

nus = [0.1,0.01,0.5,1,1.2]

final_x = gradientDescent(-2, nus)


# For finding learning rate vs covergence remove less than 10 from convergence criteria. And then uncomment below.


# learning_rate_list = [i/10 for i in range(-40,40,2)]
# final_x,first_5, iter_num = gradientDescent(-2, learning_rate_list)
# plt.plot(learning_rate_list,iter_num)
# plt.show()
# final_x = gradientDescent(-2, nus)