import sklearn.cluster as cl
import statistics as st
import random as rd
from matplotlib import pyplot as plt

num_of_elements = 40
num_of_clusters = 7


array = []
x_array = []
y_array = []
for i in range(0,num_of_elements):
    array.append([rd.randint(0,num_of_elements),rd.randint(0,num_of_elements)])
print("Массив:\n",array,"\n")

for i in range(0,num_of_elements):
    x_array.append(array[i][0])
    y_array.append(array[i][1])
plt.scatter(x_array,y_array)
plt.title("Оригінал")
plt.show()

claster_array = cl.KMeans(n_clusters=num_of_clusters, random_state=0).fit(array)
claster_labels = claster_array.labels_
claster_centers = claster_array.cluster_centers_
print("маркування кластерів:\n", claster_labels, "\n")
print("центри кластерів:\n", claster_centers)

for i in range(0,num_of_clusters):
    tmp = []
    tmp2 = []
    tmp3 = []
    for k in range(0, len(claster_labels)):
        if(claster_labels[k]==i):
            tmp.append(array[k])
            tmp2.append(array[k][0])
            tmp3.append(array[k][1])
    print("\nКластер " + str(i) +":\n", tmp)
    print("X значення:", tmp2)
    print("Y значення:", tmp3)
    print("мінімальне X:", min(tmp2))
    print("максимальне X:", max(tmp2))
    print("Середнє X:", st.mean(tmp2))
    print("Медіанне X:", st.median(tmp2))
    print("Мода x:", st.mode(tmp2))
    print("мінімальне Y:", min(tmp3))
    print("максимальне Y:", max(tmp3))
    print("Середнє Y:", st.mean(tmp3))
    print("Медіанне Y:", st.median(tmp3))
    print("Мода Y:", st.mode(tmp3))

    plt.scatter(tmp2, tmp3)
plt.title(str(num_of_clusters)+" кластерів")
plt.show()