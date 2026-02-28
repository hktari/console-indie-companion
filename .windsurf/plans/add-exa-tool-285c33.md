# Plan: Add Exa Search Tool to Real-Time Agent via REST API

This plan outlines the steps to add a web search tool to the real-time speech agent using the Exa REST API.

1.  **Get Exa API Key**: I will ask you for your Exa API key and for instructions on how to store it securely, for example, as an environment variable.

2.  **Create a new tools module:** I will create a new file `src/tools/search.py` to house the web search functionality. This will keep the tool-related code organized and separate from the main agent logic.

3.  **Implement the search tool:** In `src/tools/search.py`, I will add a function `exa_search(query: str)` that will:
    *   Read the Exa API key from the specified storage (e.g., environment variable).
    *   Make a POST request to the `https://api.exa.ai/search` endpoint with the query.
    *   Process the API response and return the search results.

4.  **Integrate the tool with the agent:** I will modify `src/voice/realtime.py` to import and use the new search tool. This will involve:
    *   Adding logic to the `_handle_server_event` method to recognize when the agent's response indicates a need to perform a search.
    *   Calling the `exa_search` function with the query from the agent's response.
    *   Passing the search results back to the agent for synthesis into the conversation.

5.  **Update the agent's prompt:** I will update the system prompt in `src/prompts/tunic_companion.py` to inform the agent about the new `search` tool. The prompt will instruct the agent to use the tool when the user asks a question and the existing knowledge base does not provide a definitive answer. This will enable the agent to effectively use the new capability.
