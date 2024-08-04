import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def get_device():
    """Determina si se debe usar CPU o GPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_image(img_path):
    """Carga la imagen desde la ruta especificada."""
    return Image.open(img_path)

def load_model_and_tokenizer(model_id, bnb_cfg):
    """
    Carga el modelo y el tokenizador desde Hugging Face Hub con la configuración de cuantización especificada.

    :param model_id: ID del modelo en Hugging Face Hub.
    :param bnb_cfg: Configuración de Bits and Bytes.
    :return: El modelo y el tokenizador cargados.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return model, tokenizer

def generate_answer(model, tokenizer, image, prompt):
    """
    Genera una respuesta usando el modelo para la imagen y el prompt proporcionados.

    :param model: El modelo cargado.
    :param tokenizer: El tokenizador cargado.
    :param image: La imagen cargada.
    :param prompt: La pregunta o prompt.
    :param device: Dispositivo a usar (CPU o GPU).
    :return: Respuesta generada por el modelo.
    """
    
    answer = model.answer_question(image, prompt, tokenizer)
    return tokenizer.decode(answer, skip_special_tokens=True)

def get_bnb_config():
    """
    Configuración de Bits and Bytes.

    :return: Configuración de Bits and Bytes.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["mm_projector", "vision_model"],
    )

def load_conversational_model(model_id):
    """
    Carga un modelo de conversación desde Hugging Face Hub.

    :param model_id: ID del modelo en Hugging Face Hub.
    :return: El modelo y el tokenizador cargados.
    """
    conversational_model = AutoModelForCausalLM.from_pretrained(model_id)
    conversational_tokenizer = AutoTokenizer.from_pretrained(model_id)
    return conversational_model, conversational_tokenizer

def converse(model, tokenizer, conversation_history, user_input):
    """
    Genera una respuesta en una conversación.

    :param model: El modelo cargado.
    :param tokenizer: El tokenizador cargado.
    :param conversation_history: El historial de la conversación.
    :param user_input: La entrada del usuario.
    :return: Respuesta generada por el modelo.
    """
    conversation_history += f"\nUser: {user_input}\nBot: "
    inputs = tokenizer(conversation_history, return_tensors="pt")#.input_ids
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    #outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    conversation_history += response
    return response, conversation_history