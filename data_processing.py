import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from plot_helpers import plot_3d

# Convert file data into a pandas dataframe
def get_data(file: str = 'Housing.csv') -> pd.DataFrame:
  print('Retrieving Data\n')
  return pd.read_csv(file)

# Reduce number of labels by rounding all prices down to nearest million dollars
def group_data(df: pd.DataFrame) -> pd.DataFrame:
  #Floor divide all data by 1 million 
  df['price'] = df['price'].apply(lambda x: int(x // 1e6))
  print(df)
  return df

# Transform the data into a useable form
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
  print('Preparing Data\n')
  le = LabelEncoder()

  df.mainroad = le.fit_transform(df.mainroad)
  df.guestroom = le.fit_transform(df.guestroom)
  df.basement = le.fit_transform(df.basement)
  df.hotwaterheating = le.fit_transform(df.hotwaterheating)
  df.airconditioning = le.fit_transform(df.airconditioning)
  df.prefarea = le.fit_transform(df.prefarea)
  df.furnishingstatus = le.fit_transform(df.furnishingstatus)

  return df

# Split data into testing and training data (80% training, 20% testing)
def split_data(dataframe: pd.DataFrame) -> tuple:
  print('Splitting Data\n')
  X = dataframe.values[:,1:]
  Y = dataframe.values[:,0]

  train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)

  return train_x, train_y, test_x, test_y

# Normalize data
def normalize(data: pd.DataFrame) -> tuple:
  print('Normalizing\n')
  scaler = StandardScaler()
  scaler.fit(data)

  return scaler, scaler.transform(data)

def preprocess() -> tuple:
  print('Preprocessing data\n')
  raw_data = get_data()
  grouped_data = group_data(raw_data)
  prepared_data = prepare_data(grouped_data)
  return split_data(prepared_data)

def preprocess_raw() -> tuple:
  print('Preprocessing (raw)\n')
  raw_data = get_data()
  prepared_data = prepare_data(raw_data)
  return split_data(prepared_data)

def plot_original_data() -> None:
  dataframe = get_data()
  prep_data = prepare_data(dataframe)
  data, labels, _, _ = split_data(prep_data)
  plot_3d(data, labels)

if __name__ == '__main__':
  plot_original_data()