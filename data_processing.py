import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from plot_helpers import plot_learning_curve
from model_template import Model


def get_data(file: str = 'Housing.csv') -> pd.DataFrame:
  return pd.read_csv(file)

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
  le = LabelEncoder()
  le.fit_transform(df.mainroad)
  le.fit_transform(df.guestroom)
  le.fit_transform(df.basement)
  le.fit_transform(df.hotwaterheating)
  le.fit_transform(df.airconditioning)
  le.fit_transform(df.prefarea)
  le.fit_transform(df.furnishingstatus)

  return df

def split_data(dataframe: pd.DataFrame) -> tuple:
  X = dataframe.values[:,1:-1]
  Y = dataframe.values[:,-1]

  train_x, train_y, test_x, test_y = train_test_split(X, Y, train_size=0.8, test_size=0.2)

  return train_x, train_y, test_x, test_y

def normalize(data: pd.DataFrame) -> tuple:
  scaler = StandardScaler()
  scaler.fit(data)

  return scaler, scaler.transform(data)

def visualize(model: Model, train_x, train_y):
  plt = plot_learning_curve(model, 'Learning Curve', train_x, train_y)
  plt.show()