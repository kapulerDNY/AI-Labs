from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def shift_mean(data):
    means = [np.mean(data[i:]) for i in range(len(data))]
    return means

def calc_score(data, k_list): #Используется Silhouette Score поэтому наибольше значение 6 , а не 5 если делать кластерезацию через К средних

    scores = []
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        labels = kmeans.labels_
        scores.append(silhouette_score(data, labels))
    return scores


#def elbow_method(data, max_clusters=15):
 #   distortions = []
  #  for i in range(1, max_clusters + 1):
   #     kmeans = KMeans(n_clusters=i, random_state=0)
    #    kmeans.fit(data)
     #   distortions.append(kmeans.inertia_)  # inertia_ содержит сумму квадратов расстояний


    #plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    #plt.title('Метод смещения среднего')
    #plt.show()


data = np.loadtxt("test.txt")


means = shift_mean(data)
optimal_clusters = np.argmax(means) + 1


k_list = np.arange(2, 15)
scores = calc_score(data, k_list)


kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(data) # методом смещения среднего (shift_mean) используется для определения оптимального числа кластеров (как по заданию), потом используем  K-средних
labels = kmeans.labels_



plt.figure(1)
plt.scatter(data[:, 0], data[:, 1])
plt.title('1. Вихідні точки на площині')




plt.figure(2)
plt.bar(np.arange(2, 15), scores)
plt.xlabel('2 - 15')
plt.title('3. Бар діаграмма score(number of clusters)')



plt.figure(3)

centers = kmeans.cluster_centers_

plt.scatter(data[:, 0], data[:, 1])


for center in centers:
    plt.scatter(centers[:, 0], centers[:, 1], marker="x", c="red")

plt.title('2. Центри кластерів (k-cредних)') #Добавил рисунок т.к кластерзацию данных я произвожу через к средних ниже код для метода зсува среднего
plt.show()

plt.figure(4)
optimal_means = shift_mean(data)
optimal_clusters_shift = np.argmax(optimal_means) + 1


plt.scatter(data[:, 0], data[:, 1])

kmeans_shift = KMeans(n_clusters=optimal_clusters_shift, random_state=0).fit(data)
centers_shift = kmeans_shift.cluster_centers_

for center in centers_shift:
    plt.scatter(center[0], center[1], marker="x", c="red")


plt.title('2. Центри кластерів (метод зсуву середнего)')
plt.show()


plt.figure(5)


for k in range(optimal_clusters):
    points = data[labels == k]

    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    y_min = points[:, 1].min()
    y_max = points[:, 1].max()
    plt.fill_between([x_min, x_max], [y_min, y_min], [y_max, y_max],
                     color=plt.cm.Spectral(k / (len(np.unique(labels)) - 1)), alpha=0.1)

    plt.plot([x_min, x_max], [y_min, y_min], color="black", linewidth=0.5)
    plt.plot([x_min, x_max], [y_max, y_max], color="black", linewidth=0.5)
    plt.plot([x_min, x_min], [y_min, y_max], color="black", linewidth=0.5)
    plt.plot([x_max, x_max], [y_min, y_max], color="black", linewidth=0.5)

plt.title('4. Кластеризованные данные с областями кластеризации')
plt.legend()
plt.show()



plt.figure(6)

for k in range(optimal_clusters):
    points = data[labels == k]
    plt.scatter(points[:, 0], points[:, 1])
plt.title('4. Кластеризованные данные с областями кластеризации')

plt.show()


#elbow_method(data)


