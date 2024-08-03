from config import get_device, load_image, load_model_and_tokenizer, generate_answer, get_bnb_config

def main():
    img_path = "C:/Users/yera_/Documents/Areas/En proceso/Qualentum-Proyectos/ProspectAI/ProspectAI/prospectos/test/paracetamol/IMG_5049.JPG"
    prompt = ('Examine the medical leaflet in the image and list the most important details related to: '
              '- What the medicine is. '
              '- What the medicine is used for. '
              '- Precautions before taking it. '
              '- Way to take it. '
              '- Adverse effects of the drug. '
              '- Conservation of the drug. '
              '- Additional Information. '
              'Finally, translate into Spanish.')

    device = get_device()
    print(f"Using device: {device}")

    image = load_image(img_path)
    bnb_cfg = get_bnb_config()
    model_id = "qresearch/llama-3-vision-alpha-hf"

    model, tokenizer = load_model_and_tokenizer(model_id, bnb_cfg)

    try:
        answer = generate_answer(model, tokenizer, image, prompt)
        print("Model answer:", answer)
    except Exception as e:
        print("Error during model inference:", str(e))

if __name__ == "__main__":
    main()
