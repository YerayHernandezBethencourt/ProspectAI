import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig

def get_device():
    """Determina el dispositivo a usar (GPU o CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_path: str, device: str):
    """
    Carga el modelo y el tokenizador desde la ruta especificada.
    
    :param model_name: Nombre del modelo a cargar.
    :param device: Dispositivo en el que se cargará el modelo.
    :return: El modelo y el tokenizador cargados.
    """
    try:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        model = model.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error al cargar el modelo o tokenizador: {e}")
        raise
    """try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
        model = model.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error al cargar el modelo o tokenizador: {e}")
        raise"""

def analyze_image(image_path: str, model, tokenizer, question: str):
    """
    Analiza una imagen utilizando el modelo y el tokenizador proporcionados.
    
    :param image_path: Ruta a la imagen que se analizará.
    :param model: Modelo de Transformers.
    :param tokenizer: Tokenizador de Transformers.
    :param question: Pregunta para el análisis.
    :return: Respuesta generada por el modelo.
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return None
    
    msgs = [{'role': 'user', 'content': question}]
    try:
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,  # Si es False, se usará la búsqueda por haz (beam search) por defecto
            temperature=0.1
        )
        return res
    except Exception as e:
        print(f"Error durante la inferencia del modelo: {e}")
        return None