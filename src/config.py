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
