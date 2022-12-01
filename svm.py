from sklearn.svm import SVC
from data_processing import preprocess, normalize
from sklearn.metrics import accuracy_score
from plot_helpers import plot_learning_curve

if __name__ == '__main__':
  svm = SVC(kernel='rbf')

  train_x, train_y, test_x, test_y = preprocess()

  scaler, train_data = normalize(train_x)

  svm.fit(train_x, train_y)

  test_x = scaler.transform(test_x)

  predicted_labels = svm.predict(test_x)

  accuracy = accuracy_score(test_y, predicted_labels)
  print(accuracy)
  