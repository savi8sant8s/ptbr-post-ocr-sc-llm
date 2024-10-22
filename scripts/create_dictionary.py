import pandas as pd
from symspellpy import SymSpell

# For Symspell and Norvig

# Ground truth of Bluche, Flor and Puigcerver are the same
predictions = pd.read_csv('train_data/bluche.csv')
ground_truth = predictions['output'].to_list()

synthetics_predictions = pd.read_csv('train_data/synthetic_prompts.csv')
ground_truth_synthetic = synthetics_predictions['output'].to_list()

symspell = SymSpell()

texts = ground_truth + ground_truth_synthetic

symspell.create_dictionary(texts)

sorted_words = sorted(symspell.words.items(), key=lambda x: x[1], reverse=True)

with open('train_data/dictionary.txt', 'w') as f:
    for word, count in sorted_words:
        f.write(f'{word} {count}\n')