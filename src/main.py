import streamlit as st
from config import get_device, load_image, load_model_and_tokenizer, generate_answer, get_bnb_config
from conversation import setup_conversation_chain, conversation
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def main():
    st.title("ProspectAI")

    img_path = st.text_input("Ruta de la imagen del prospecto:", value="C:/Users/yera_/Documents/Areas/En proceso/Qualentum-Proyectos/ProspectAI/ProspectAI/prospectos/test/paracetamol/paracetamol.JPG")
    prompt = ('Examine the medical leaflet in the image and list the most important details related to: '
              '- What the medicine is. '
              '- What the medicine is used for. '
              '- Precautions before taking it. '
              '- Way to take it. '
              '- Adverse effects of the drug. '
              '- Conservation of the drug. '
              '- Additional Information. '
              'Finally, translate the results into Spanish.'
              'Only the information extracted.')

    if st.button("Procesar imagen"):
        device = get_device()
        st.write(f"Usando dispositivo: {device}")

        image = load_image(img_path)
        bnb_cfg = get_bnb_config()
        model_id = "qresearch/llama-3-vision-alpha-hf"
        model, tokenizer = load_model_and_tokenizer(model_id, bnb_cfg)
        
        try:
            extracted_info = generate_answer(model, tokenizer, image, prompt)
            st.write("Información extraída del modelo:", extracted_info)
            st.session_state["extracted_info"] = extracted_info
        except Exception as e:
            st.write("Error durante la inferencia del modelo:", str(e))

    if "extracted_info" in st.session_state:
        # Configurar el modelo de conversación y el prompt
        llm = Ollama(model="llama3.1:8b")
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", f"""Eres un asistente virtual que ayudará con las preguntas referentes al medicamento del cual tienes esta información: {st.session_state['extracted_info']}\
                    \n Tus conversaciones podrían seguir estos ejemplos:
                        {{
                            "role": "user",
                            "content": "¿Puedo tomar este medicamento si estoy embarazada?"
                            }},
                        {{
                            "role": "assistant",
                            "content": "Si está embarazada, debe consultar a su médico antes de tomar Paracetamol cinfa. Su médico evaluará los riesgos y beneficios de tomar este medicamento durante el embarazo."
                            }},
                        {{
                            "role": "user",
                            "content": "¿Qué debo hacer si experimento una reacción alérgica?"
                            }},
                        {{
                            "role": "assistant",
                            "content": "Si experimenta una reacción alérgica, como dificultad para respirar, hinchazón de la cara, labios, lengua o garganta, debe buscar atención médica de inmediato. Interrumpa el uso del medicamento y consulte a su médico."
                            }}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = setup_conversation_chain(llm, prompt_template)
        
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_input = st.text_input("Escribe tu pregunta:", key="user_input")

        if user_input:
            response, updated_history = conversation(user_input, chain, st.session_state["chat_history"])
            st.session_state["chat_history"] = updated_history
            if response:
                st.write(f"Bot: {response}")

if __name__ == "__main__":
    main()
