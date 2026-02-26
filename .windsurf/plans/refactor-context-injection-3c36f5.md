# Refactor Context Injection Logic

This plan refactors the VLM context injection to be more efficient. Instead of injecting on every frame, context will only be injected when the user speaks or when a critical in-game event occurs. Each context injection will be coupled with RAG queries to provide the AI with grounded information. The number of scenes for context and for RAG queries will be independently configurable.

## Implementation Steps

1.  **`src/main.py`**
    *   Remove the continuous `context_mgr.flush_to_voice(voice)` call from the main loop.
    *   Update critical event handling (death, low health) to call a new `flush_latest_to_voice` method in the `ContextManager`.

2.  **`src/context/manager.py`**
    *   Implement `get_recent_context(num_scenes_context: int, num_scenes_rag: int) -> str`. This method will:
        *   Retrieve the last `num_scenes_context` from history for the description.
        *   Perform RAG queries for each of the last `num_scenes_rag` scenes.
        *   Aggregate and de-duplicate the RAG results.
        *   Return a single formatted string containing both the scene descriptions and the aggregated RAG results.
    *   Implement `flush_latest_to_voice(voice_session)`. This will format and inject only the most recent scene and its associated RAG results, for use in critical events.

3.  **`src/voice/realtime.py`**
    *   Update `VoiceSession.__init__` to accept the `ContextManager` instance.
    *   In `_handle_server_event`, on user speech completion (`conversation.item.input_audio_transcription.completed`), read the new config values and call `context_mgr.get_recent_context()` to inject the result.

4.  **`src/voice/session_config.json`**
    *   Add two new keys:
        *   `"num_scenes_for_context": 5`
        *   `"num_scenes_for_rag": 3`
