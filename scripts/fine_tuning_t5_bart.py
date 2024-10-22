import torch
 
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BartTokenizer, 
    BartForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

dataset = load_dataset('csv', data_files='prompts/puigcerver_train.csv')
dataset_full = dataset['train'].train_test_split(shuffle=True, test_size=0.1)
dataset_train, dataset_valid = dataset_full['train'], dataset_full['test']

#model_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
model_name =  'adalbertojunior/bart-base-portuguese'
tokenizer = BartTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    all_wrong = examples['input']
    all_wrong = [f"Corrija: {text}" for text in all_wrong]
    all_correct = examples['output']
    model_inputs = tokenizer(
        all_wrong, 
        max_length=128,
        truncation=True,
        padding='max_length'
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            all_correct, 
            max_length=128,
            truncation=True,
            padding='max_length'
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = dataset_train.map(
    preprocess_function, 
    batched=True,
    num_proc=8
)
tokenized_valid = dataset_valid.map(
    preprocess_function, 
    batched=True,
    num_proc=8
)

model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

out_dir = 'bart_portuguese_puigcerver'
batch_size = 16
epochs = 2

training_args = TrainingArguments(
    output_dir=out_dir,               
    num_train_epochs=epochs,              
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,                
    weight_decay=0.03,               
    logging_dir=out_dir,            
    logging_steps=10,
    evaluation_strategy='steps',    
    save_steps=500,                 
    eval_steps=500,                 
    load_best_model_at_end=True,     
    save_total_limit=10,
    report_to='tensorboard',
    learning_rate=0.0001,
    dataloader_num_workers=8,
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=tokenized_train,       
    eval_dataset=tokenized_valid,
)
history = trainer.train()

model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
