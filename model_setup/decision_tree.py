from sklearn.tree import DecisionTreeClassifier
from model_template import Model
from data_processing import preprocess

#TODO setup decision tree to be imported
if __name__ == '__main__':
  train_x, train_y, test_x, test_y = preprocess()

  tree = Model('Decision Tree', DecisionTreeClassifier(max_depth=300, max_leaf_nodes=50), train_x, train_y, test_x, test_y)

  tree.train()
  tree.print_metrics(predict_arg='micro')