import os
import streamlit as st
from LlmService import LlmService

# initialize LLM service

service = LlmService()

query = "What is the capital of Portugal?"

result = service.query(query)

print(result)
