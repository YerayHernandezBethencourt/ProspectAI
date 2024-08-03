import os
from PIL import Image
from config import get_device, load_model_and_tokenizer, analyze_image
from transformers import AutoTokenizer, AutoModel

def main():
    device = get_device()
    print(f"Usando dispositivo: {device}")
    
    model_path = 'C:/Users/yera_/.cache/huggingface/hub/models--openbmb--MiniCPM-Llama3-V-2_5/snapshots/41008ff9ed95a75a4ee26b7f4bf50e7903e1645e'
    image_path = "C:/Users/yera_/Documents/Areas/En proceso/Qualentum-Proyectos/ProspectAI/ProspectAI/prospectos/test/paracetamol/IMG_5049.JPG"
    prompt = (
        'Examina el prospecto médico de la imagen y enumera los detalles más importantes relacionados con: '
        '- Qué es el medicamento. '
        '- Para qué se utiliza el medicamento. '
        '- Precauciones antes de tomarlo. '
        '- Forma de tomarlo. '
        '- Efectos adversos del fármaco.'
        '- Conservación del fármaco. '
        '- Información adicional.'
    )
    
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_path)
    try:
        answer = analyze_image(model, tokenizer, image_path, prompt, device)
        print("Respuesta del modelo:", answer)
    except Exception as e:
        print("Error durante la inferencia del modelo:", str(e))
if __name__ == '__main__':
    main()
    