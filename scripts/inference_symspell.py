import pandas as pd
from symspellpy import SymSpell, Verbosity
import re

sym_spell = SymSpell()

dictionary_path = 'experiments/third/train_data/dictionary.txt'
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

df = pd.read_csv("experiments/third/test_data/azure.csv")
inputs = df["prediction"].tolist()

predictions = []

for input_text in inputs:
    if not isinstance(input_text, str):
        predictions.append("")
        continue
    if input_text == "SKIP":
        predictions.append("SKIP")
        continue
    tokens = re.findall(r'\w+(?:-\w+)*|[^\w\s]', input_text, re.UNICODE)
    predictions_per_input = []
    
    for token in tokens:
        if re.match(r'\w+', token):
            suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2, transfer_casing=True)
            if suggestions:
                predictions_per_input.append(suggestions[0].term)
            else:
                predictions_per_input.append(token)
        else:
            predictions_per_input.append(token)
    
    corrected_text = ""
    for token in predictions_per_input:
        if corrected_text and re.match(r'\w', token):
            corrected_text += " " + token
        else:
            corrected_text += token
    predictions.append(corrected_text)

df_predictions = pd.DataFrame(predictions, columns=["prediction"])

df_predictions.to_csv("experiments/corrections/symspell/azure.csv", index=False)
