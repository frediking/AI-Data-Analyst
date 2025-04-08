from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import logging

logger = logging.getLogger(__name__)

def init_chat_state():
    """Initialize chat-specific session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_initialized' not in st.session_state:
        st.session_state.chat_initialized = False

def create_chat_prompt():
    """Create prompt template for chat"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Act as a data analysis expert. Using the data context below, answer the user's question.
        
        Data Context:
        {context}
        
        User Question:
        {question}
        
        Your Response:
        """
    )

def initialize_chat():
    """Initialize chat components"""
    try:
        model = Ollama(model="llama2")
        prompt = create_chat_prompt()
        chain = LLMChain(llm=model, prompt=prompt)
        st.session_state.chat_initialized = True
        return chain
    except Exception as e:
        logger.error(f"Chat initialization failed: {str(e)}")
        raise RuntimeError(f"Failed to initialize chat: {str(e)}")

# def generate_response(chain: LLMChain, question: str, df_context: str) -> str:
#     """Generate chat response"""
#     try:
#         response = chain.run({
#             "context": df_context,
#             "question": question
#         })
#         return response.strip()
#     except Exception as e:
#         logger.error(f"Failed to generate response: {str(e)}")
#         return "I'm sorry, I encountered an error processing your question."