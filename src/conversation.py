import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm=Ollama(model="llama3.1:8b")

def conversacion(contexto):
    st.title("Chat")
    prompt = f"""Eres un asistente virtual que ayudará con las preguntas referentes al medicamento del cual tienes esta información{contexto}. 
    Las querys deben ser respondidas con amabilidad y certeza plena. 
    Algunos ejempos del contexto y su conversación:
        {
        "contexto": "El prospecto médico de la imagen proporciona información detallada sobre Paracetamol cinfa, un medicamento del grupo de analgésicos y antipiréticos. Este medicamento está indicado para el tratamiento sintomático del dolor leve o moderado y de la fiebre en adultos y adolescentes mayores de 15 años y con un peso superior a 50 kg. Antes de tomarlo, debe consultar a su médico si está tomando otros medicamentos, si padece alguna enfermedad hepática, renal o cardíaca, si está embarazada o en período de lactancia, entre otras condiciones. La dosis recomendada para adultos y adolescentes mayores de 15 años y con un peso superior a 50 kg es de 1 comprimido cada 6-8 horas, según necesidad, sin exceder los 3 comprimidos en 24 horas. Los efectos adversos pueden incluir reacciones alérgicas, problemas hepáticos y trastornos hematológicos, entre otros. Debe conservarse en un lugar seco y a temperatura ambiente, protegido de la luz. Para más información, consulte a su médico o farmacéutico.",
        },
        {
          "role": "user",
          "content": "¿Puedo tomar este medicamento si estoy embarazada?"
        },
        {
          "role": "assistant",
          "content": "Si está embarazada, debe consultar a su médico antes de tomar Paracetamol cinfa. Su médico evaluará los riesgos y beneficios de tomar este medicamento durante el embarazo."
        },
        {
          "role": "user",
          "content": "¿Qué debo hacer si experimento una reacción alérgica?"
        },
        {
          "role": "assistant",
          "content": "Si experimenta una reacción alérgica, como dificultad para respirar, hinchazón de la cara, labios, lengua o garganta, debe buscar atención médica de inmediato. Interrumpa el uso del medicamento y consulte a su médico."
        }        {
          "role": "user",
          "content": "¿Puedo tomar este medicamento si estoy embarazada?"
        },
        {
          "role": "assistant",
          "content": "Si está embarazada, debe consultar a su médico antes de tomar Paracetamol cinfa. Su médico evaluará los riesgos y beneficios de tomar este medicamento durante el embarazo."
        },
        {
          "role": "user",
          "content": "¿Qué debo hacer si experimento una reacción alérgica?"
        },
        {
          "role": "assistant",
          "content": "Si experimenta una reacción alérgica, como dificultad para respirar, hinchazón de la cara, labios, lengua o garganta, debe buscar atención médica de inmediato. Interrumpa el uso del medicamento y consulte a su médico."
        },
        
        {
          "contexto": "El prospecto médico de la imagen proporciona información detallada sobre Vitamina B12 Farmasierra, un medicamento que contiene cianocobalamina (vitamina B12) como principio activo. Se utiliza para tratar y prevenir la anemia perniciosa y otras deficiencias de vitamina B12, así como en casos de dietas inadecuadas (vegetarianos estrictos) y ciertas condiciones médicas que dificultan la absorción de esta vitamina. Antes de tomarlo, debe consultar a su médico si tiene alergia a la cianocobalamina u otras cobalaminas, si padece alguna enfermedad hereditaria del nervio óptico, si tiene deficiencia de ácido fólico o si está embarazada o en período de lactancia. La dosis y la duración del tratamiento deben ser determinadas por el médico, basándose en la severidad de la deficiencia y la respuesta del paciente al tratamiento. Los efectos adversos pueden incluir reacciones alérgicas, trastornos gastrointestinales, y raramente, problemas hematológicos. Debe conservarse en un lugar seco y a temperatura ambiente, protegido de la luz. Para más información, consulte a su médico o farmacéutico."
        },
        {
          "role": "user",
          "content": "¿Puedo tomar este medicamento si tengo deficiencia de ácido fólico?"
        },
        {
          "role": "assistant",
          "content": "Si tiene deficiencia de ácido fólico, debe consultar a su médico antes de tomar Vitamina B12 Farmasierra. La combinación de deficiencia de vitamina B12 y ácido fólico puede requerir un ajuste específico en el tratamiento."
        },
        {
          "role": "user",
          "content": "¿Qué debo hacer si experimento trastornos gastrointestinales?"
        },
        {
          "role": "assistant",
          "content": "Si experimenta trastornos gastrointestinales, consulte a su médico. Podría ser necesario ajustar la dosis o la forma de administración del medicamento."
        },
        {
            "contexto": "El prospecto médico de la imagen proporciona información detallada sobre el medicamento Alprazolam, que es un fármaco perteneciente a la familia de los benzodiazepínicos. El medicamento se utiliza comúnmente para tratar síntomas de ansiedad y estrés. Antes de tomar este medicamento, es importante ser consciente de las precauciones, como no utilizarlo si se tiene alergia a Alprazolam o si se está tomando otros medicamentos que puedan interactuar con él. También se indica que el medicamento debe tomarse exactamente según las instrucciones del médico y no se debe sobredosis ni suspender repentinamente.En cuanto a los efectos adversos, el prospecto menciona posibles reacciones secundarias como somnolencia, debilidad muscular, mareo, visión borrosa, fatiga, irritabilidad, ansiedad aumentada, depresión, confusión, problemas de memoria, alteraciones del lenguaje, problemas de coordinación, temblor, y en casos raros, pensamientos suicidas o intentos de suicidio. Es fundamental informar al médico de cualquier efecto adverso observado.Para la conservación del medicamento, se sugiere almacenarlo en un lugar seco y fresco, lejos de la luz directa y de los niños. Además, el prospecto incluye una sección de información adicional que podría contener detalles sobre interacciones medicamentosas, uso durante el embarazo y la lactancia, y otros aspectos importantes relacionados con el uso del medicamento."},
            {
              "role": "user",
              "content": "¿Puedo tomar este medicamento si estoy embarazada?"
            },
            {
              "role": "assistant",
              "content": "El prospecto sugiere que antes de tomar Alprazolam durante el embarazo, se debe consultar con un médico. El uso de benzodiazepinas durante el embarazo puede estar asociado con riesgos para el feto."
            },
            {
              "role": "user",
              "content": "¿Qué debo hacer si me olvido de tomar una dosis?"
            },
            {
              "role": "assistant",
              "content": "Si olvida tomar una dosis de Alprazolam, tómela tan pronto como lo recuerde. Si ya es casi la hora de la siguiente dosis, omita la dosis olvidada y tome la siguiente dosis a la hora habitual. No tome una dosis doble para compensar la dosis olvidada."
            },
            {
                "contexto": "El prospecto médico de la imagen proporciona una variedad de detalles importantes sobre el medicamento. El medicamento se identifica como 'Hidroferol 0,266 mg' y se describe como un medicamento que contiene ácido hialurónico. Se utiliza para tratar síntomas relacionados con la artritis reumatoide, incluyendo dolor articular, rigidez y limitación funcional. Antes de tomar el medicamento, se recomienda informar a un profesional de la salud si se tiene alergia a ácido hialurónico o cualquier otro componente del medicamento, si se está tomando otros medicamentos o si se está embarazada o amamantando. La forma de administración es oral, y se debe tomar con agua. Los efectos adversos potenciales incluyen diarrea, estreñimiento, dolor abdominal, vómito, dolor de cabeza, insomnio y dolor muscular. El medicamento debe conservarse en un lugar seco y alejado de la luz directa y calor excesivo. Además, se proporciona información adicional sobre la composición del medicamento, su uso en niños y adultos, y las interacciones con otros medicamentos."
            },
            {
              "role": "user",
              "content": "¿Es seguro tomar este medicamento si tengo problemas estomacales?"
            },
            {
              "role": "assistant",
              "content": "Si tiene problemas estomacales, debe informar a su médico antes de tomar Hidroferol. Algunos efectos secundarios incluyen diarrea y dolor abdominal, por lo que su médico puede decidir si es adecuado para usted."
            },
            {
              "role": "user",
              "content": "¿Puede este medicamento interactuar con otros medicamentos que estoy tomando?"
            },
            {
              "role": "assistant",
              "content": "Sí, el prospecto sugiere que Hidroferol puede interactuar con otros medicamentos. Es importante informar a su médico sobre todos los medicamentos que está tomando para evitar posibles interacciones."
            }
    """
    bot_description = st.text_area("Describa su pregunta:", value=prompt)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", bot_description),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "input"),
        ]
    )
    
    chain = prompt_template | llm
    
    chat = chat(chain)
    
    user_input = st.text_input("Escribe tu pregunta:", key="user_input")

