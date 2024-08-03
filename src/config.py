import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

def get_device():
    """Determina el dispositivo a usar (GPU o CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_path: str):
    """
    Carga el modelo y el tokenizador desde un directorio local.

    :param model_path: Ruta al directorio local del modelo.
    :return: El modelo y el tokenizador cargados.
    """
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer
    """try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
        model = model.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error al cargar el modelo o tokenizador: {e}")
        raise"""

def analyze_image(model, tokenizer, image_path: str, question: str, device: str):
    """
    Realiza la inferencia en la imagen proporcionada con la pregunta dada.

    :param model: El modelo cargado.
    :param tokenizer: El tokenizador cargado.
    :param image_path: Ruta a la imagen.
    :param question: Pregunta para el modelo.
    :param device: Dispositivo a usar (CPU o GPU).
    :return: Respuesta generada por el modelo.
    """
    image = Image.open(image_path)
    inputs = tokenizer(question, return_tensors="pt").to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(**inputs)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer