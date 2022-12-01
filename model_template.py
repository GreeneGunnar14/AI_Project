#Import libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score
from plot_helpers import plot_learning_curve
#Import preprocessed data

#Create model, define training and predicting on data
class Model:
  def __init__(self, Name, Estimator, train_x, train_y, test_x, test_y):
    self.name = Name
    self.estimator = Estimator
    self.train_x = train_x
    self.train_y = train_y
    self.test_x = test_x
    self.test_y = test_y

  def train(self) -> None:
    print(f'Training Model ({self.name})\n')
    self.estimator.fit(X=self.train_x, y=self.train_y)

  def get_metrics(self, predict_arg: str=None) -> dict:
    print(f'Predicting with model ({self.name})\n')
    scores = {'Accuracy': 0, 'Precision': 0, 'Recall': 0}
    labels = self.estimator.predict(self.test_x)
    accuracy = accuracy_score(self.test_y, labels)
    if not predict_arg:
      precision = precision_score(self.test_y, labels)
      recall = recall_score(self.test_y, labels)
    else:
      precision = precision_score(self.test_y, labels, average=predict_arg)
      recall = recall_score(self.test_y, labels, average=predict_arg)
    scores['Accuracy'] = accuracy
    scores['Precision'] = precision
    scores['Recall'] = recall

    return scores

  def visualize_learning_curve(self):
    plt = plot_learning_curve(self.estimator, f'{self.name} Learning Curve', self.train_x, self.train_y)
    plt.show()

  def save_learning_curve(self):
    plt = plot_learning_curve(self.estimator, f'{self.name} Learning Curve', self.train_x, self.train_y)
    plt.savefig(f'{self.name}_learning_curve.png')