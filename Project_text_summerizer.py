
from transformers import pipeline, set_seed
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset
#AutoModelForSeq2SeqLM is a class from the Hugging Face Transformers library that automatically loads a sequence-to-sequence (seq2seq) model architecture. It simplifies the process of selecting and initializing models for tasks like summarization, translation, or text generation
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,DataCollatorForSeq2Seq,TrainingArguments,Trainer
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch

nltk.download("punkt")

metric = evaluate.load("accuracy")

#C:/Users/Saw/AppData/Local/Programs/Python/Python312/python.exe -m pip install py7zr

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(device)

model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

dataset_samsum = load_dataset("samsum",trust_remote_code=True)

# print(dataset_samsum)

# print(dataset_samsum["train"]["dialogue"][1])
# print(dataset_samsum["train"][1]["summary"])
split_lengths = [len(dataset_samsum[split])for split in dataset_samsum]

# print(f"split lengths : {split_lengths}")
# print(f"features : {dataset_samsum["train"].column_names}")
# print("\ndialouge: ")
# print(dataset_samsum["test"][1]["dialogue"])
# print("\nsummary")
# print(dataset_samsum["test"][1]["summary"])

#convert to tokenizer
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["dialogue"],max_length=1024,truncation=True)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch["summary"],max_length=128,truncation=True)
    
    return {
        'input_ids' : input_encodings['input_ids'],
        'attention_mask' : input_encodings['attention_mask'],
        'labels' : target_encodings['input_ids']
    }
dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features,batched= True)
# print(dataset_samsum)
# print(dataset_samsum_pt)

# print(dataset_samsum_pt["train"])
# print(dataset_samsum_pt["train"]["input_ids"][1])

# print(dataset_samsum_pt["train"]["attention_mask"][1])
# print(dataset_samsum_pt["train"]["labels"][1])
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer,model=model_pegasus)
#TrainingArguments object for fine-tuning a model using the Hugging Face  transformers library
trainer_args = TrainingArguments(
    output_dir='pegasus-samsum',
    #Sets the number of times the entire training dataset will be passed through the model (1 epoch in this case).
    num_train_epochs=1,
    # Gradually increases the learning rate for the first 500 steps to help the model stabilize during the initial stages of training.
    warmup_steps=500,
    #Sets the batch size for evaluation. Each batch will contain one example per device (e.g., one GPU or CPU core).
    per_device_eval_batch_size=1,
    #Sets the batch size for training. Each batch will contain one example per device.
    per_device_train_batch_size=1,
    #Adds a regularization term to the loss to prevent overfitting. A weight decay of 0.01 penalizes large model weights during optimization.
    weight_decay=0.01,
    #Logs the training progress (e.g., loss, metrics) every 10 steps.
    logging_steps=10,
    #Defines when to run evaluations. Here, it will evaluate the model after a specified number of steps (eval_steps).
    evaluation_strategy='steps',
    #Runs the evaluation loop every 500 steps during training.

    eval_steps=500,
    #Specifies how often the model's checkpoints should be saved. Here, it saves checkpoints every 1,000,000 steps (effectively disables frequent saving).
    save_steps=1e6,
    # Accumulates gradients over 16 steps before updating the model's weights. This is useful when using small batch sizes, as it effectively increases the "virtual batch size" to 16 \times \text{per_device_train_batch_size}.Gradient Update in Training:Normally, after processing each batch of data, the model computes gradients (how much to adjust each weight in the model) and updates the weights.Instead of updating the weights after every batch, gradient_accumulation_steps delays the weight update until gradients have been accumulated over multiple batches.In this case, gradients are accumulated over 16 batches before updating the weights.If the per_device_train_batch_size=1 (1 example per batch) and gradient_accumulation_steps=16, the model effectively behaves as if the batch size were 1x16= 16 This means the model sees 16 examples before making a single weight update.Processes a mini-batch of size 1.Computes gradients for that batch and accumulates them (adds them to a "gradient buffer").Does not update the weights immediately.After 16 such steps (when gradient_accumulation_steps=16)The accumulated gradients are averaged.The model updates the weights using the averaged gradients.The gradient buffer is cleared, and the process repeats.
    gradient_accumulation_steps=16
)

trainer = Trainer(model= model_pegasus,
                  args=trainer_args,
                  tokenizer=tokenizer,
                  data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["train"],
                  eval_dataset=dataset_samsum_pt["validation"])

trainer.train()