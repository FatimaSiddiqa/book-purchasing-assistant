from fastapi import FastAPI, Request, HTTPException
from langserve import add_routes
from graph import graph_app, State  # Update this import to match your main file name
import logging
from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add routes for the graph_app
add_routes(app, graph_app, path="/chat")

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        request_data = await request.json()
        logger.debug(f"Received request: {request_data}")
        
        input_data = request_data.get('input', {})
        messages = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in input_data.get('messages', [])
        ]
        state = State(
            messages=messages,
            book_name=input_data.get('book_name', ''),
            iteration_count=0,
            action=''
        )
        logger.debug(f"Initial state: {state}")

        # Process state through the graph
        final_output = None
        for output in graph_app.stream(state):
            logger.debug(f"Graph output: {output}")
            if isinstance(output, dict):
                if "perform_action" in output:
                    action_output = output["perform_action"]
                    if "messages" in action_output and action_output["messages"]:
                        final_output = action_output["messages"][-1].content
                        state["book_name"] = action_output.get("book_name", state["book_name"])
                elif "end" in output:
                    end_output = output["end"]
                    if "messages" in end_output and end_output["messages"]:
                        final_output = end_output["messages"][-1].content
                        state["book_name"] = end_output.get("book_name", state["book_name"])
            if final_output:
                break

        if final_output:
            logger.info(f"Returning response: {final_output}")
            return {"output": final_output, "book_name": state["book_name"]}
        else:
            logger.warning("No valid response generated")
            return {"output": "I'm sorry, I couldn't generate a response. Please try again.", "book_name": state["book_name"]}

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return {"output": "An error occurred while processing your request. Please try again later.", "book_name": state["book_name"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)