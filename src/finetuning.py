import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from datasets import load_dataset, load_metric

# 1. Cargar modelo y tokenizador
model_name = "openbmb/MiniCPM-Llama3-V-2_5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 2. Cargar datasets
train_dataset = load_dataset('json', data_files='train_anotations.json', split='train')
val_dataset = load_dataset('json', data_files='val_anotations.json', split='train')
test_dataset = load_dataset('json', data_files='test_anotations.json', split='train')

# 3. Definir función de preprocesamiento
def preprocess_function(examples):
    # Verifica el tipo de `examples`
    if isinstance(examples, dict):
        # Imprime el tipo y ejemplo de datos para depuración
        print(f"Tipo de examples: {type(examples)}")
        print(f"Ejemplo de datos: {examples}")

        # Asegúrate de que `examples` tenga las claves correctas
        if 'conversations' in examples:
            conversations = examples['conversations']
            
            # Procesa los datos
            inputs = []
            for conversation in conversations:
                if isinstance(conversation, dict):
                    content = conversation.get("content", "")
                    inputs.append(content)
                else:
                    inputs.append("")
                    
            return {"inputs": inputs}
        else:
            raise KeyError("'conversations' key is missing in the examples")
    else:
        raise TypeError(f"Expected a dict but got {type(examples)}")

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 4. Configurar argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    eval_steps=500,
    logging_dir='./logs',
)

# 5. Definir métricas
rouge = load_metric("rouge")
bleu = load_metric("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    bleu_score = bleu.compute(predictions=[pred.split() for pred in decoded_preds], references=[[label.split()] for label in decoded_labels])
    result["bleu"] = bleu_score["bleu"]
    return result

# 6. Definir Trainer y entrenar el modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo finetuneado
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
