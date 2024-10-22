import pandas as pd

df_1 = pd.read_csv("train_data/bluche.csv")
df_2 = pd.read_csv("train_data/flor.csv")
df_3 = pd.read_csv("train_data/puigcerver.csv")
df_4 = pd.read_csv("train_data/synthetic_prompts.csv")

df_new = pd.concat([df_1, df_2, df_3, df_4], ignore_index=True)

df_new = df_new.sample(frac=1).reset_index(drop=True)

df_new.to_csv("combined_prompts.csv", index=False)