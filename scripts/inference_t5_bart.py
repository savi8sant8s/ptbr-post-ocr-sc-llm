from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
import pandas as pd
import torch
import time

model_path = "models/ptt5_puigcerver"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_csv("prompts/puigcerver_test.csv")

predictions = pd.DataFrame([], columns=["prediction"])

def do_correction(text, model, tokenizer):
    input_text = f"Corrija: {text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=128,
        padding='max_length',
        truncation=True
    )
    inputs = inputs.to(device)
    corrected_ids = model.generate(
        inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )
    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=True
    )
    return corrected_sentence 

outputs = []

start = time.time()

for index, row in df.iterrows():
    input_text = row['input']
    corrected_sentence = do_correction(input_text, model, tokenizer)
    print(f'Time: {time.time() - start:.2f}s')
    outputs.append(corrected_sentence)
    
predictions["prediction"] = outputs
predictions.to_csv("predictions/ptt5_puigcerver.csv", index=False)

print(f"Total time: {time.time() - start:.2f}s")