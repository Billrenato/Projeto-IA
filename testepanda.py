import pandas as pd

df = pd.read_csv("dataset/labels.csv")
print(df[df['label'].isnull() | (df['label'].str.strip() == "")])