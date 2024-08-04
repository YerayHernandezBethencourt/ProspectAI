from config import get_device, load_image, load_model_and_tokenizer, generate_answer, get_bnb_config, load_conversational_model, converse
from conversation import conversacion as conver
def main():
    img_path = "C:/Users/yera_/Documents/Areas/En proceso/Qualentum-Proyectos/ProspectAI/ProspectAI/prospectos/test/paracetamol/paracetamol.JPG"
    prompt = ('Examine the medical leaflet in the image and list the most important details related to: '
              '- What the medicine is. '
              '- What the medicine is used for. '
              '- Precautions before taking it. '
              '- Way to take it. '
              '- Adverse effects of the drug. '
              '- Conservation of the drug. '
              '- Additional Information. '
              'Finally, translate the results into Spanish.'
              'Only the information extracted')

    device = get_device()
    print(f"Using device: {device}")

    image = load_image(img_path)
    bnb_cfg = get_bnb_config()
    model_id = "qresearch/llama-3-vision-alpha-hf"

    model, tokenizer = load_model_and_tokenizer(model_id, bnb_cfg)

    try:
        extracted_info = generate_answer(model, tokenizer, image, prompt)
        print("Model answer:", extracted_info)
    except Exception as e:
        print("Error during model inference:", str(e))
        
    # Conversational model
    #conversational_model_id = "meta-llama/Meta-Llama-3.1-8B"
    #conversational_model, conversational_tokenizer = #load_conversational_model(conversational_model_id)
    chat = conver(extracted_info)
    #conversation_history = f"Information extracted from image: {extracted_info}\n"
    
    print("Puede hacer preguntas sobre la información obtenida. Escriba 'exit' para finalizar la conversación.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response, conversation_history = converse(conversational_model, conversational_tokenizer, conversation_history, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
