# Advanced Voice Assistant with Integrated CLAP Agent, MCP, and A2A

<p align="center">
  <img src="media/GITCLAP.png" alt="CLAP Framework Logo" width="700"/>
</p>

This project demonstrates a sophisticated, voice-controlled AI assistant built using the Google Agent Development Kit (ADK). Jarvis can manage Google Calendar, interact with Gmail via a Model Context Protocol (MCP) server, and delegate complex knowledge-based queries (including Retrieval Augmented Generation - RAG) to a custom agent built with the CLAP framework, communicating via the Agent-to-Agent (A2A) protocol.



<p align="center">
  <img src="media/PIP_CLAP.png" alt="CLAP Pip Install Example" width="700"/>
</p>

## Architecture Overview

The system comprises three main, independently running server components:

1.  **ADK Voice Agent (Jarvis - FastAPI UI & ADK Core)**:
    *   The primary user interface, providing voice input/output via a web UI (FastAPI + WebSockets).
    *   Core agent logic built with Google ADK (`Agent` class, `Runner`).
    *   Uses a Gemini model (e.g., `gemini-1.5-flash-latest`) for its main reasoning.
    *   **Responsibilities:**
        *   Handles real-time voice interaction (streaming input/output).
        *   Directly manages Google Calendar operations using local ADK tools.
        *   Connects to the "Gmail MCP Server" using `MCPToolset` to send emails and fetch recent emails.
        *   Connects to the "CLAP A2A Server" using a custom ADK tool (acting as an A2A client) to delegate complex queries, especially those requiring RAG over a specific knowledge base (e.g., the Holbox AI PDF).
    *   Runs typically on `http://localhost:8000`.

2.  **Gmail MCP Server (Starlette & MCP SDK)**:
    *   A dedicated server exposing Gmail functionalities (send email, fetch recent emails) via the Model Context Protocol (MCP).
    *   Built using `mcp.server.fastmcp.FastMCP` and served via Starlette.
    *   Communicates with the ADK Voice Agent over Server-Sent Events (SSE).
    *   Requires Gmail App Password for SMTP/IMAP access.
    *   Runs typically on `http://localhost:8001` (or the port configured in `gmail_mcp.py`).

3.  **CLAP A2A Server (FastAPI & CLAP Framework)**:
    *   Hosts a custom agent built using the **CLAP (Cognitive Layer Agent Package)** framework.
    *   This CLAP agent is configured for RAG capabilities over a specified document (e.g., a PDF about Holbox AI).
    *   Exposes its CLAP agent's functionality to other agents via the Agent-to-Agent (A2A) protocol.
    *   Implemented as a FastAPI application for direct control over HTTP responses.
    *   Runs typically on `http://localhost:9999`.

**Conceptual Flow:**

[User (Voice/Web UI)] <---- (WebSocket) ----> [ADK Voice Agent (Jarvis - FastAPI @ 8000)]
| | |
(ADK Tools) | (MCPToolset via SSE) (A2A Client Tool via HTTP)
| | |
v v v
[Google Calendar API] [Gmail MCP Server @ 8001] [CLAP A2A Server @ 9999]
(SMTP/IMAP to Gmail) (Hosts CLAP RAG Agent)
(ChromaDB, LLM Service)


## Key Features

*   **Voice-Controlled Interface:** Interactive and responsive user experience using Google ADK.
*   **Modular Agent Capabilities:**
    *   **Calendar Management:** Directly handled by the ADK agent.
    *   **Gmail Integration:** Via a dedicated, reusable MCP server.
    *   **Advanced Knowledge/RAG:** Delegated to a specialized CLAP agent via A2A, allowing complex document querying (e.g., about Holbox AI).
*   **Inter-Agent Communication:** Demonstrates A2A protocol for service-like calls between the ADK agent and the CLAP agent.
*   **MCP Tool Usage:** Shows ADK `MCPToolset` connecting to an SSE-based MCP server.
*   **Custom Agent Framework (CLAP):** Leverages the CLAP framework for building the RAG-capable backend agent.
*   **Retrieval Augmented Generation (RAG):** The CLAP agent can answer questions based on a PDF document using a vector store (ChromaDB).


## Prerequisites

*   **Python:** Version 3.10 recommended.
*   **Conda (or other virtual environment manager):** `holbox` environment used.
*   **API Keys & Credentials:**
    *   Google API Key (for Gemini, via AI Studio).
    *   Gmail App Password (for Gmail MCP Server).
    *   API Key for CLAP Agent's LLM (e.g., Groq or another Google API Key if using Gemini for CLAP).
    *   Google Calendar API `credentials.json`.
*   **PDF Document for RAG:** Placed in `HOLBOXATHON/clap_a2a_integration/` (e.g., `holbox_ai_info.pdf`).

## Setup

1.  **Clone the Project.**
2.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n holbox python=3.10 -y
    conda activate holbox
    ```
3.  **Install Dependencies:**
    *   ADK Voice Agent: `cd ADK/adk-voice-agent && pip install -r requirements.txt && cd ../..`
    *   CLAP A2A & Gmail MCP: Ensure dependencies like `fastapi`, `uvicorn`, `starlette`, `python-dotenv`, `mcp`, `a2a-sdk`, `requests`, `chromadb`, `sentence-transformers`, `pypdf`, and the SDK for your CLAP agent's LLM are included, likely in the ADK `requirements.txt` or a root project `requirements.txt`. If CLAP is a local package, install it via `pip install -e path/to/clap_framework_root`.
4.  **Configure Environment Variables (`.env` files):**
    *   `ADK/adk-voice-agent/.env`: `GOOGLE_API_KEY`, `CLAP_A2A_SERVER_URL="http://localhost:9999"`
    *   `mcps/.env`: `SMTP_USERNAME`, `SMTP_PASSWORD`
    *   `clap_a2a_integration/.env`: API key for CLAP agent's LLM (e.g., `GOOGLE_API_KEY` or `GROQ_API_KEY`).
5.  **Setup Google Calendar API:** Run `python ADK/adk-voice-agent/setup_calendar_auth.py` and authorize.
6.  **Place RAG Document:** Put your PDF (e.g., `holbox_ai_info.pdf`) in `clap_a2a_integration/`.

## Running the System

Run each server in a separate terminal, in order:

1.  **Gmail MCP Server:**
    ```bash
    conda activate holbox
    python mcps/gmail_mcp.py
    ```
    (Runs on port 8001 or 5000 as configured)

2.  **CLAP A2A Server (with RAG):**
    ```bash
    conda activate holbox
    python clap_a2a_integration/run_clap_a2a_server_fastapi_style.py
    ```
    (Runs on port 9999. First run ingests the PDF.)

3.  **ADK Voice Agent (Jarvis):**
    ```bash
    conda activate holbox
    cd ADK/adk-voice-agent/app
    uvicorn main:app --reload
    ```
    (Runs on port 8000)

## How to Use

1.  Open `http://localhost:8000` in your browser.
2.  Interact with Jarvis using voice or text:
    *   **Calendar:** "What's my schedule for today?"
    *   **Gmail:** "Send an email to my_friend@example.com, subject Hello, body Just checking in!"
    *   **CLAP Agent (Holbox AI Info):** "Ask the CLAP agent: What are the key services of Holbox AI?"

## Key Technologies Used (Top 10)

Python, Google Agent Development Kit (ADK), FastAPI, CLAP Framework, WebSockets, Google Gemini API, A2A (Agent-to-Agent) Protocol & SDK, MCP (Model Context Protocol) & SDK, ChromaDB, JavaScript (for UI).

