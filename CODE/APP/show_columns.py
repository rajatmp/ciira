import pandas as pd

df = pd.read_csv("Train_Test_Network.csv")
df.columns = df.columns.str.strip().str.lower()

print("\n📋 Columns in Train_Test_Network.csv:")
for col in df.columns:
    print(f"- '{col}'")
