import os
import streamlit as st
from LlmService import LlmService
from Chat import Chat


st.title('LLM Service')

chat = Chat()


# initialize LLM service
service = LlmService()


def perform_query(query_string):
    chat.add_chat_history('user', query_string)
    answer = service.query(query_string)
    chat.add_chat_history('assistant', answer)

query = st.chat_input( placeholder="Ask a question...")


if (query is None):
    st.stop()

perform_query(query)



chat_history = chat.get_chat_history()

# st.write(chat.get_chat_history())


for message in chat_history:
    with st.chat_message(message['role']):
        st.write(message['message'])

