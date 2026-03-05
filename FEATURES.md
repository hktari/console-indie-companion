# Features and Functional Scope: Tunic Voice Companion

A voice-enabled AI companion for the game **TUNIC** that provides contextual assistance, lore discussions, and real-time gameplay reactions by watching the player's screen.

## 1. Functional Requirements

### 1.1 Computer Vision & Scene Understanding
- **Live Screen Capture**: Capture active windows (specifically PS Remote Play or Chiaki-ng) using X11 and `mss`.
- **Scene Analysis**: Utilize Gemini 2.5 Flash to analyze gameplay screenshots every 3 seconds.
- **Game State Extraction**: Identify current location, player health, visible enemies, UI elements, and active tasks from visual data.
- **Replay Mode**: Support for testing using pre-captured screenshot sequences without a live game connection.

### 1.2 Knowledge Retrieval (RAG)
- **Wiki Integration**: Scrape and index Tunic-specific knowledge from community wikis (e.g., Fandom).
- **Vector Search**: Use ChromaDB to retrieve relevant gameplay hints and lore based on the current visual context.
- **Context Synthesis**: Combine VLM-derived scene data with RAG-retrieved knowledge to provide a comprehensive state update for the AI.

### 1.3 Voice Interaction
- **Speech-to-Speech**: Low-latency conversation using OpenAI's Realtime API (~400ms latency).
- **Voice Activity Detection (VAD)**: Automatically detect when the user is speaking and when to respond.
- **Session Management**: Handle automatic session rotation (at 55 minutes) to stay within API limits while preserving recent context.
- **Microphone/Speaker Integration**: Direct interaction through the user's local audio hardware.

### 1.4 Companion Personality & Logic
- **Non-Assistant Persona**: Operates as a "friend hanging out" rather than a formal assistant—casual, enthusiastic, and reactive.
- **Graduated Hint System**: 
  - **Level 1**: Vague nudges.
  - **Level 2**: Specific directions/references.
  - **Level 3**: Full spoilers (prefaced with a warning).
- **Reactive Engagement**: The AI primarily waits for user input but can react to critical system events (e.g., player death or low health).
- **Spoiler Protection**: Guardrails to prevent revealing unencountered mechanics or locations.

### 1.5 Utility & Monitoring
- **Cost Tracking**: Real-time logging of API usage (Gemini and OpenAI) with hourly cost estimations.
- **Configurable Performance**: Adjustable capture intervals and model selection (Flash vs. Flash-Lite) to balance latency and cost.

## 2. Technical Scope

### 2.1 Supported Environment
- **Operating System**: Linux (X11-based) required for window capture tools.
- **Input Sources**: PS Remote Play, Chiaki-ng, or local replay directory.
- **Hardware**: Standard PC microphone and headphones.

### 2.2 Integration Points
- **Google Gemini API**: Vision-language modeling for scene understanding.
- **OpenAI Realtime API**: High-fidelity, low-latency voice interaction.
- **Tunic Wiki (Fandom)**: Source for game-specific knowledge base.

### 2.3 Out of Scope (POC Phase)
- **Proactive Commentary**: AI does not offer unsolicited advice (except for critical health/death events).
- **Persistent Long-Term Memory**: Conversation history is limited to the current session (max 60 mins).
- **Mobile Support**: Requires a desktop environment for capture and processing.
- **Multi-Game Support**: Currently hard-coded with Tunic-specific prompts and data.
