import os
import streamlit as st
from LlmService import LlmService
from Chat import Chat


st.title('LLM Service')


# initialize Chat service
if 'chat'  in st.session_state:
    chat = st.session_state['chat']
else:
    # Set up chat service
    chat = Chat()
    st.session_state['chat'] = chat


# initialize LLM service
if 'service'  in st.session_state:
    service = st.session_state['service']
else:
    service = LlmService()
    st.session_state['service'] = service

# initialize cache hit counter
if 'cache_hit_count' not in st.session_state:
    st.session_state.cache_hit_count = 0


def perform_query(query_string = None):
    query_string = st.session_state.query
    chat.add_chat_history('user', query_string)
    answer = service.query(query_string)
    chat.add_chat_history('assistant', answer)
    st.session_state.cache_hit_count = service.get_cache_hit_count()

chat_history = chat.get_chat_history()


with st.sidebar:
    st.write("Stats")
    st.write("Cache hits: " + str(st.session_state.cache_hit_count))


with st.container(height=500):

    for message in chat_history:
        with st.chat_message(message['role']):
            st.write(message['message'])

query = st.chat_input( placeholder="Ask a question...", on_submit=perform_query, key="query")

