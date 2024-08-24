import streamlit as st
import requests
from langchain_core.messages import HumanMessage, AIMessage

st.title("Book Purchasing Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "book_name" not in st.session_state:
    st.session_state.book_name = ""

if not st.session_state.book_name:
    book_name = st.text_input("Enter the name of the book:")
    if book_name:
        st.session_state.book_name = book_name
        st.rerun()

st.write(f"Current book: {st.session_state.book_name}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input prompt for user
prompt = st.chat_input("Ask a question about the book, or type 'change book to [new book name]' to switch books")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send request to backend
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"input": {"messages": st.session_state.messages, "book_name": st.session_state.book_name}}
        )
        
        if response.status_code == 200:
            response_data = response.json()
            assistant_response = response_data.get("output", "")
            new_book_name = response_data.get("book_name")
            if new_book_name:
                st.session_state.book_name = new_book_name
            if assistant_response:
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
            else:
                st.error("Received an empty response from the backend.")
        else:
            st.error(f"Failed to get response from backend. Status code: {response.status_code}")
            st.error(f"Response content: {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

    st.rerun()