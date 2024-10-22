from ngram import NGram
import pandas as pd

df_1 = pd.read_csv('train_data/synthetic_prompts.csv')
output_1 = df_1['output'].tolist()

df_2 = pd.read_csv('train_data/real_prompts/puigcerver.csv')
output_2 = df_2['output'].tolist()

output = output_1 + output_2

words_ = []

for text in output:
    words_.extend(text.split())

ngram = NGram(set(words_), N=3)

def correct_text(text):
    words = text.split()
    correct_words = []
    for word in words:
        if word in output:
            correct_words.append(word)
        else:
            correct_words.append(ngram.find(word))
    return ' '.join(correct_words)

optical_models = ['bluche', 'flor', 'puigcerver', 'pero','ltu_main','ltu_ensemble','litis','demokritos', 'azure']

for model in optical_models:
    data = pd.read_csv(f'test_data/{model}.csv')
    df_pred = pd.DataFrame([], columns=['prediction'])

    for index, row in data.iterrows():
        print(index)
        text = row['prediction']
        prediction = pd.DataFrame([correct_text(text)], columns=['prediction'])
        df_pred = pd.concat([df_pred, prediction], ignore_index=True)
        df_pred.to_csv(f'corrections/ngram/{model}.csv', index=False)