# Semantic Cache Demo

This is a Python app using [Streamlit](https://streamlit.io/) for a UI

## Installation

Install dependencies with pipenv

```sh
pipenv install
```

## Set up env
Copy `.env.example` to `.env` and set your OpenAI API key.

## Run the app in Streamlit
Run streamlit with:

```sh
pipenv run python -m streamlit run ./src/main.py
```

The cache is set up knowing the capital of Canada, France, and Germany. You can ask about the capital of these countries and see the cache hit. Other queries will hit the LLM and then will be cached for subsequent requests.
