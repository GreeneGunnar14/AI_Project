from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from data_processing import preprocess
from plot_helpers import plot_elbow_method,  plot_silhouette_method, plot_3d

def test_clusters(train_x):
  clusters = range(2, 21)

  inertia_values = []
  silhouette_scores = []

  for i in tqdm(clusters):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(train_x)
    inertia_values.append(kmeans.inertia_)
    score = silhouette_score(train_x, kmeans.labels_)
    silhouette_scores.append(score)
  plot_elbow_method(inertia_values)
  plot_silhouette_method(silhouette_scores)

if __name__ == '__main__':
  train_x, train_y, test_x, test_y = preprocess()

  num_clusters=15

  kmeans = KMeans(n_clusters=num_clusters)

  kmeans.fit(train_x, train_y)

  plot_3d(train_x, kmeans.labels_)

  