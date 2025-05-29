import streamlit as st

# import setup first to get dotenv running
import app.setup
from app.Agent import Agent
from app.Chat import Chat



st.title('LLM Service')


# initialize Chat agent
if 'chat'  in st.session_state:
    chat = st.session_state['chat']
else:
    # Set up chat agent
    chat = Chat()
    st.session_state['chat'] = chat


# initialize Agent
if 'agent'  in st.session_state:
    agent = st.session_state['agent']
else:
    agent = Agent()
    st.session_state['agent'] = agent

# initialize cache hit counter
if 'cache_hit_count' not in st.session_state:
    st.session_state.cache_hit_count = 0


def perform_query(query_string = None):
    query_string = st.session_state.query
    chat.add_chat_history('user', query_string)
    answer = agent.query(query_string)
    chat.add_chat_history('assistant', answer)
    st.session_state.cache_hit_count = agent.get_cache_hit_count()

chat_history = chat.get_chat_history()


with st.sidebar:
    st.write("Stats")
    st.write("Cache hits: " + str(st.session_state.cache_hit_count))


with st.container(height=500):

    for message in chat_history:
        with st.chat_message(message['role']):
            st.write(message['message'])

query = st.chat_input( placeholder="Ask a question...", on_submit=perform_query, key="query")

