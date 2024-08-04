import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def setup_conversation_chain(llm, prompt_template):
    """Configura el modelo de conversación con el `llm` y el `prompt_template` proporcionado."""
    chain = prompt_template | llm
    return chain


def conversation(user_input, chain, chat_history):
    """Maneja la interacción con el modelo de conversación y actualiza el historial de chat."""
    if user_input:
        # Add user message to the history
        chat_history.append(HumanMessage(content=user_input))
        
        # Prepare inputs for the chain
        inputs = {
            "input": user_input,
            "chat_history": chat_history
        }
        
        try:
            response = chain(inputs)
            response_text = response["output"]  # Modify this if response structure is different
            
            # Add AI message to the history
            chat_history.append(AIMessage(content=response_text))
            return response_text, chat_history
        
        except Exception as e:
            st.write(f"Error en la generación de la respuesta: {e}")
            return "Hubo un error en la generación de la respuesta.", chat_history
    return None, chat_history