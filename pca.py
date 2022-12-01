from sklearn.decomposition import PCA
from data_processing import preprocess, normalize

if __name__ == '__main__':
  train_x, train_y, test_x, test_y = preprocess()

  scaler, train_data = normalize(train_x)

  pca = PCA(13)
  pca.fit_transform(train_data)

