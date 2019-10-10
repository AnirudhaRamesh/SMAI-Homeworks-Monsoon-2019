import numpy as np
import random
import matplotlib.pyplot as plt

classa = []
classb = []
xlista = []
ylista = []
xlistb = []
ylistb = []

for i in range(100):
    a = random.uniform(-1,1)
    b = random.uniform(-1,1)
    c = 1;

    group = []
    group.append(a)
    group.append(b)
    group.append(c)

    if (i%2 == 0):
        classa.append(group)
        xlista.append(a)
        ylista.append(b)
    else:
        classb.append(group)
        xlistb.append(a)
        ylistb.append(b)

w1 = [1,1,0]
w2 = [-1,-1,0]
w3 = [0,0.5,0]
w4 = [1,-1,5]
w5 = [1,1,0.3]

finala1 = []
finalb1 = []
accuracy1 = 0
finala2 = []
finalb2 = []
accuracy2 = 0
finala3 = []
finalb3 = []
accuracy3 = 0
finala4 = []
finalb4 = []
accuracy4 = 0
finala5 = []
finalb5 = []
accuracy5 = 0

#----------------------------------------

for i in classa:
    resulta1 = []
    resulta1 = [a*b for a,b in zip(w1,i)]
    totala1 = 0
    for ele in range(0, len(resulta1)): 
        totala1 = totala1 + resulta1[ele] 
    finala1.append(totala1)
    
for i in classb:
    resultb1 = []
    resultb1 = [a*b for a,b in zip(w1,i)]
    totalb1 = 0
    for ele in range(0, len(resultb1)):
        totalb1 = totalb1 + resultb1[ele]
    finalb1.append(totalb1)



for i in finala1:
    if (i > 0):
        accuracy1 += 1

for i in finalb1:
    if (i <= 0):
        accuracy1 += 1

print(accuracy1)

#-------------------------------------------

for i in classa:
    resulta2 = []
    resulta2 = [a*b for a,b in zip(w2,i)]
    totala2 = 0
    for ele in range(0, len(resulta2)): 
        totala2 = totala2 + resulta2[ele] 
    finala2.append(totala2)
    
for i in classb:
    resultb2 = []
    resultb2 = [a*b for a,b in zip(w2,i)]
    totalb2 = 0
    for ele in range(0, len(resultb2)):
        totalb2 = totalb2 + resultb2[ele]
    finalb2.append(totalb2)



for i in finala2:
    if (i > 0):
        accuracy2 += 1

for i in finalb2:
    if (i <= 0):
        accuracy2 += 1

print(accuracy2)

#-------------------------------------------

for i in classa:
    resulta3 = []
    resulta3 = [a*b for a,b in zip(w3,i)]
    totala3 = 0
    for ele in range(0, len(resulta3)): 
        totala3 = totala3 + resulta3[ele] 
    finala3.append(totala3)
    
for i in classb:
    resultb3 = []
    resultb3 = [a*b for a,b in zip(w3,i)]
    totalb3 = 0
    for ele in range(0, len(resultb3)):
        totalb3 = totalb3 + resultb3[ele]
    finalb3.append(totalb3)



for i in finala3:
    if (i > 0):
        accuracy3 += 1

for i in finalb3:
    if (i <= 0):
        accuracy3 += 1

print(accuracy3)

#-------------------------------------------

for i in classa:
    resulta4 = []
    resulta4 = [a*b for a,b in zip(w4,i)]
    totala4 = 0
    for ele in range(0, len(resulta4)): 
        totala4 = totala4 + resulta4[ele] 
    finala4.append(totala4)
    
for i in classb:
    resultb4 = []
    resultb4 = [a*b for a,b in zip(w4,i)]
    totalb4 = 0
    for ele in range(0, len(resultb4)):
        totalb4 = totalb4 + resultb4[ele]
    finalb4.append(totalb4)



for i in finala4:
    if (i > 0):
        accuracy4 += 1

for i in finalb4:
    if (i <= 0):
        accuracy4 += 1

print(accuracy4)

#-------------------------------------------

for i in classa:
    resulta5 = []
    resulta5 = [a*b for a,b in zip(w5,i)]
    totala5 = 0
    for ele in range(0, len(resulta5)): 
        totala5 = totala5 + resulta5[ele] 
    finala5.append(totala5)
    
for i in classb:
    resultb5 = []
    resultb5= [a*b for a,b in zip(w5,i)]
    totalb5 = 0
    for ele in range(0, len(resultb5)):
        totalb5 = totalb5 + resultb5[ele]
    finalb5.append(totalb5)



for i in finala5:
    if (i > 0):
        accuracy5 += 1

for i in finalb5:
    if (i <= 0):
        accuracy5 += 1

print(accuracy5)

#-------------------------------------------


plt.scatter(xlista, ylista, color = 'orange')
plt.scatter(xlistb, ylistb, color = 'green')


plt.plot([1, 0.5, -0.5, -1], [-1, -0.5, 0.5, 1],color = 'black')
plt.plot([0, -1, -0.5, 1], [5, 4, 4.5, 6],color = 'red')
plt.plot([1, -1, -0.3, 0], [-1.3, 0.7, 0, -0.3],color = 'blue')

plt.show()


