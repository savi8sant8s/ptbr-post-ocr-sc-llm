import pandas as pd
from autocorrect import Speller

dictionary = open('train_data/dictionary.txt', 'r')

lines = dictionary.readlines()

word_frequency_dict = {}

for line in lines:
    word, freq = line.split()
    word_frequency_dict[word] = int(freq)

spell = Speller(nlp_data=word_frequency_dict)


df = pd.read_csv("test_data/azure.csv")

predictions = []

for index, row in df.iterrows():
    input_text = row['prediction']
    if not isinstance(input_text, str):
        predictions.append("")
        continue
    if input_text == "SKIP":
        predictions.append("SKIP")
        continue
    print(index)
    predictions.append(spell(input_text))

df_predictions = pd.DataFrame(predictions, columns=["prediction"])

df_predictions.to_csv("corrections/norvig/azure.csv", index=False)