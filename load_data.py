import pandas as pd

def load_data_from_csv(filename):
  data = pd.read_csv(filename)
  return data
