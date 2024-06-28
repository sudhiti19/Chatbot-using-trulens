import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from trulens_eval import TruChain, Feedback, Huggingface, Tru

hugs = Huggingface()
tru = Tru()
load_dotenv()

# Load environment variables from .streamlit/secrets.toml
cohere_api_key = os.getenv("COHERE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

template = """You are a chatbot having a conversation with a human.
        {chat_history}
        Human: {human_input}
        Chatbot:"""
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")
llm = ChatCohere(cohere_api_key=cohere_api_key)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

# # Question/answer relevance between overall question and answer.
# f_relevance = Feedback(llm.relevance).on_input_output()

# # Moderation metrics on output
# f_hate = Feedback(llm.moderation_hate).on_output()
# f_violent = Feedback(llm.moderation_violence, higher_is_better=False).on_output()
# f_selfharm = Feedback(llm.moderation_selfharm, higher_is_better=False).on_output()
# f_maliciousness = Feedback(llm.maliciousness_with_cot_reasons, higher_is_better=False).on_output()

chain_recorder = TruChain(
    chain
)

st.title("Contextual Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Record with TruLens
        with chain_recorder as recording:
            full_response = chain.run(prompt)
        message_placeholder = st.empty()
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})

tru.run_dashboard()