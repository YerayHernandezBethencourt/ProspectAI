import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Ruta al archivo JSON
data_files = {
    "train": "../anotations/train_anotations.json",
    "validation": "../anotations/val_anotations.json"
}

# Configura el modelo y el tokenizer
MODEL = "openbmb/MiniCPM-Llama3-V-2_5"
DATA = "../anotations/train_anotations.json"
EVAL_DATA = "../anotations/val_anotations.json"
LLM_TYPE = "llama3"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)

# Carga los datos
def preprocess_function(examples):
    inputs = [ex["conversations"][0]["content"] for ex in examples["data"]]
    targets = [ex["conversations"][1]["content"] for ex in examples["data"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = load_dataset("json", data_files="train_anotations.json", split="train")
eval_dataset = load_dataset("json", data_files="val_anotations.json", split="train")

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Configura los argumentos de entrenamiento
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs",
    output_dir="./finetuned_model",
    evaluation_strategy="steps",
    save_steps=10_000,
    eval_steps=10_000,
    logging_steps=200,
    do_train=True,
    do_eval=True,
)

# Configura el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Entrena el modelo
trainer.train()
