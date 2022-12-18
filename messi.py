import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy

img = plt.imread("messi.png")

width = img.shape[0]
height = img.shape[1]

img = img.reshape(width*height,4)
kmeans = KMeans(n_clusters=8).fit(img)

labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

img2 = numpy.zeros_like(img)

for i in range(len(img2)):
	img2[i] = clusters[labels[i]]

img2 = img2.reshape(width,height,4)

plt.imshow(img2)
plt.show()