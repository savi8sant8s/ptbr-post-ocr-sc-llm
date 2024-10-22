import pandas as pd
from ast import literal_eval
import random
import noisocr

df = pd.read_csv("https://raw.githubusercontent.com/lplnufpi/essay-br/main/extended-corpus/extended_essay-br.csv")

input = []
output = []

for index, row in df.iterrows():
    if len(row['essay']) < 5:
        continue
    essay = literal_eval(row['essay'][1:-1])
    print(f"Processing essay {index+1}/{len(df)}")
    lines = []

    for line in essay:
        lines.extend(noisocr.sliding_window_with_hyphenization(line))
    for index2, prompt in enumerate(lines):
        if (len(prompt) < 3):
            continue
        output = noisocr.simulate_annotation(prompt)
        text_error = noisocr.simulate_errors(output, random.randint(2, 6), seed=42)
        input.append(text_error)
        output.append(output)
        
df_output = pd.DataFrame({
    'input': input,
    'output': output,
})

df_output.to_csv('synthetic_prompts.csv', index=False, header=True)
