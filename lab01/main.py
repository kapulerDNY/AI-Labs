import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score


data = np.loadtxt("test.txt")

# 1. методом зсуву середнего визначає кількість кластерів розбиття
def meanShift(data):
    bandwidth = MeanShift(bandwidth=None)
    bandwidth.fit(data)
    return len(np.unique(bandwidth.labels_)), bandwidth.cluster_centers_

# 2. Оценка score
def calc_score(data, max_clusters=15):
    scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        scores.append(score)
    return scores

# 3. проводить кластеризацію методом k-середних з оптимальною кількістю кластерів
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

# 1. методом зсуву середнего визначає кількість кластерів розбиття,
num_clusters_shift, meanshift_centers = meanShift(data)

# потім оцінити score для різних вариіантів кластеризації (кількість кластерів від 2 до 15) в функций от 2 до 15 + 1 - то есть включительно
cluster_scores = calc_score(data)

# оптимальною кількістю кластерів
optimal_num_clusters = np.argmax(cluster_scores) + 2

# проводить кластеризацію методом k-середних з оптимальною кількістю кластерів
kmeans_labels, kmeans_centers = kmeans_clustering(data, optimal_num_clusters)


plt.figure(1)
# 1. Вихідні точки на площині
plt.scatter(data[:, 0], data[:, 1], c='black', marker='o')
plt.title('Исходные точки')

plt.figure(2)
# 2. Центри кластерів (метод зсуву середнего)
plt.scatter(data[:, 0], data[:, 1], c='black', marker='o')
plt.scatter(meanshift_centers[:, 0], meanshift_centers[:, 1], c='red', marker='x')
plt.title('Центры кластеров (сдвиг среднего)')

plt.figure(3)
# 3. Бар диаграмма score(number of clusters)
plt.bar(range(2, 16), cluster_scores)
plt.title('Бар диаграмма score(number of clusters)')
plt.xlabel('Количество кластеров')


plt.figure(4)
# 4. Кластеризованные данные с областями кластеризации(k-средних)
plt.scatter(data[:, 0], data[:, 1], c=kmeans_labels, marker='o')
plt.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], c='red', marker='x')
plt.title('Кластеризованные данные с областями кластеризации (k-средних)')

plt.show()

plt.figure(5)


for k in range(optimal_num_clusters):
    points = data[kmeans_labels == k]

    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    y_min = points[:, 1].min()
    y_max = points[:, 1].max()
    plt.fill_between([x_min, x_max], [y_min, y_min], [y_max, y_max],
                     color=plt.cm.Spectral(k / (len(np.unique(kmeans_labels)) - 1)), alpha=0.1)

    plt.plot([x_min, x_max], [y_min, y_min], color="black", linewidth=0.5)
    plt.plot([x_min, x_max], [y_max, y_max], color="black", linewidth=0.5)
    plt.plot([x_min, x_min], [y_min, y_max], color="black", linewidth=0.5)
    plt.plot([x_max, x_max], [y_min, y_max], color="black", linewidth=0.5)

plt.title('4. Кластеризованные данные с областями кластеризации(Области)')
plt.legend()
plt.show()
