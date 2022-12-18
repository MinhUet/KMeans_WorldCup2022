import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy

img = plt.imread("messi.png")

width = img.shape[0]
height = img.shape[1]
dimension = img.shape[2]

# Thay đổi cụm để có màu tốt nhất
K = 4 

img = img.reshape(width*height,dimension)
kmeans = KMeans(n_clusters=K).fit(img)

labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

img2 = numpy.zeros_like(img)

for i in range(len(img2)):
	img2[i] = clusters[labels[i]]

img2 = img2.reshape(width,height,dimension)

plt.imshow(img2)
plt.show()