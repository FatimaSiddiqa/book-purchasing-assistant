import os
from typing import Annotated, List, TypedDict
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import tool
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_core.callbacks import CallbackManager

# Set API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
if not os.environ["GROQ_API_KEY"]:
    raise ValueError("GROQ_API_KEY is not set in environment variables")

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


# Initialize DuckDuckGo tool
duckduckgo_search = DuckDuckGoSearchRun()
# Initialize Tavily tool
search = TavilySearchAPIWrapper()
tavily_search = TavilySearchResults(api_wrapper=search)
# Initialize LLM
llm = ChatGroq(temperature=0,model="llama3-8b-8192")

@tool
def duckduckgo(book_name: str) -> str:
    """Returns a summary of the specified book."""
    try:
        result = duckduckgo_search.run(f"Summary of the book '{book_name}'")        
        prompt = f"""
        Based on the following information, create a well-structured summary of the book '{book_name}' in about 150 words:

        {result}

        Include the following points if available:
        1. Author's name
        2. Main characters
        3. Basic plot outline
        4. Main themes
        5. Setting (time and place)

        Format the summary in clear, separate paragraphs.
        """
        
        summary = llm.invoke(prompt)
        return summary.content
    except Exception as e:
        logger.error(f"Error in DuckDuckgo search for '{book_name}', action 'summary': {e}")
        raise

@tool
def tavily(query: str) -> str:
    """Performs a Tavily search for the given query."""
    try:
        result = tavily_search.invoke(query)
        
        # Extract book name and action from the query
        if "recommendations of books similar to" in query:
            action = "recommendations"
            book_name = query.split("similar to '")[1].split("'")[0]
        elif "reviews of the book" in query:
            action = "reviews"
            book_name = query.split("reviews of the book '")[1].split("'")[0]
        else:
            action = "other"
            book_name = query.split("regarding the book '")[1].split("'")[0]
        
        if action == 'reviews':
            format_prompt = f"""
            Based on the following information about reviews of the book '{book_name}', provide a well-structured response:

            {result}

            Provide 3 diverse reviews of the book. For each review:
            1. Provide the source (website or publication name)
            2. Give a brief summary of the review (2-3 sentences)
            3. State the overall sentiment (positive, negative, or mixed)
            4. If available, include a short quote from the review
            Format each review separately and clearly.
            """
        elif action == 'recommendations':
            format_prompt = f"""
            Based on the following information about books similar to '{book_name}', provide a well-structured response:

            {result}

            Recommend 5 books similar to '{book_name}'. For each recommendation:
            1. Provide the book title and author
            2. Briefly explain why it's similar to the original book (1-2 sentences)
            3. Mention a key theme or element that connects it to the original book
            List the recommendations clearly, numbering them from 1 to 5.
            """
        else:
            format_prompt = f"""
            Based on the following information about the book '{book_name}', provide a well-structured response:

            {result}

            Answer the following question about the book:
            {query}
            Provide a clear, concise answer in about 100 words.
            """

        formatted_response = llm.invoke(format_prompt)
        return formatted_response.content
    except Exception as e:
        logger.error(f"Error in Tavily search: {e}")
        raise


class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "add_messages"]
    book_name: str
    iteration_count: int
    action: str

graph = StateGraph(State)

def should_continue(state: State) -> bool:
    if state["iteration_count"] >= 5:
        return False
    return isinstance(state["messages"][-1], HumanMessage)

def router(state: State):
    messages = state['messages']
    if not messages:
        return {"action": "other", "iteration_count": state["iteration_count"] + 1}
    
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return {"action": "other", "iteration_count": state["iteration_count"] + 1}
    
    user_input = last_message.content.strip().lower()
        
    if user_input.startswith("change book to"):
        new_book = user_input.replace("change book to", "").strip()
        return {"action": "change_book", "book_name": new_book, "iteration_count": state["iteration_count"] + 1}
    
    prompt = f"""
    Categorize the following user request about the book '{state['book_name']}' into one of these categories: summary, reviews, recommendations, gratitude, finish or other.
    User request: {user_input}

    Respond with only one word: the category name. If the request doesn't clearly fit into summary, reviews, finish, gratitude, or recommendations, respond with 'other'.
    """

    response = llm.invoke(prompt)
    action = response.content.strip().lower()

    if user_input in ['quit', 'exit', 'bye', 'thanks', 'end']:
        return {"action": "end", "iteration_count": state["iteration_count"] + 1}
    if action in ["finish","gratitude"]:
        return {"action": "end", "iteration_count": state["iteration_count"] + 1}
    if action not in ['summary', 'reviews', 'recommendations', 'other']:
        action = 'other'

    return {"action": action, "iteration_count": state["iteration_count"] + 1, "book_name": state['book_name']}


def perform_action(state: State):
    action = state['action']
    book_name = state['book_name']
    messages = state['messages']
    user_input = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ''

    if action == 'end':
        return {"messages": messages + [AIMessage(content="Thank you for using the Book Assistant. Goodbye!")], "iteration_count": state["iteration_count"] + 1}

    if action == 'change_book':
        return {"messages": messages + [AIMessage(content=f"Book changed to '{book_name}'. What would you like to know about this book?")], "iteration_count": state["iteration_count"] + 1, "book_name": book_name}

    try:
        callback_manager = CallbackManager([])  # Create an empty CallbackManager
        if action == 'summary':
            tool_result = duckduckgo.run(book_name, callbacks=callback_manager)
        else:
            if action == 'recommendations':
                query = f"search recommendations of books similar to '{book_name}'"
            elif action == 'reviews':
                query = f"search reviews of the book '{book_name}'"
            else:
                query = f"answer the following question regarding the book '{book_name}': {user_input}"
            
            tool_result = tavily.run(query, callbacks=callback_manager)

        return {"messages": messages + [AIMessage(content=f"\n--- {action.capitalize()} for '{book_name}' ---\n{tool_result}\n")], "iteration_count": state["iteration_count"] + 1, "book_name": book_name}
    except Exception as e:
        error_message = f"Error in perform_action for book '{book_name}', action '{action}': {str(e)}"
        return {"messages": messages + [AIMessage(content=f"I apologize, I encountered an error while trying to get {action} for '{book_name}'. Error details: {error_message}. Please try again or ask about a different book or topic.")], "iteration_count": state["iteration_count"] + 1, "book_name": book_name}
    

def end(state: State):
    return state

# Add nodes to the graph
graph.add_node("router", router)
graph.add_node("perform_action", perform_action)
graph.add_node("end", end)

# Add edges
graph.add_edge("router", "perform_action")
graph.add_conditional_edges(
    "perform_action",
    should_continue,
    {
        True: "router",
        False: "end"
    }
)

# Set entry point
graph.set_entry_point("router")

# Compile the graph
graph_app = graph.compile()