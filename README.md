# Agentic book-purchasing-assistant

This application uses Langgraph agents and langchain tools to help users research a book before they buy it.

# graph.py
defines the Langraph with the agents plus Tavily and DuckDuckgo tools to search the internet

# app.py:
uses FastApi and Langserve to deploy the application

# streamlit_app.py:
A streamlit frontend for the users to interact with the application.


# How to run:
Step 01: Ensure all three files are in the same directory
Step 02: Set the environment variables (GROQ_API_KEY, TAVILY_API_KEY)
Step 03: python run app.py
Step 04: streamlit run streamlit_app.py
