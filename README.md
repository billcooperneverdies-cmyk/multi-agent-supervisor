# Multi-Agent Supervisor

A modular AI architecture with three layers:
1. **Orchestration Layer** (LangGraph) — State machine + agent routing
2. **Middleware Component Protocol (MCP) Server** — Tool interface for external APIs (GitHub, Slack)
3. **Model Routing Layer** (LiteLLM) — Unified LLM proxy with tool calling

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  LangGraph (Orchestration)                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  Router  │──▶│  Agent   │──▶│  Tool    │        │
│  │  Node    │   │  Node    │   │  Node    │        │
│  └──────────┘   └──────────┘   └──────────┘        │
│        │              │              │               │
│        └──────────────┘              │               │
│                     AgentState       │               │
└──────────────────────────────────────┼───────────────┘
                                       │ stdio
┌──────────────────────────────────────┼───────────────┐
│  MCP Server (Middleware)             │               │
│  ┌──────────┐   ┌──────────┐         │               │
│  │ github   │   │ slack    │◀──────┘               │
│  │ _create_ │   │ _send_   │                       │
│  │ _pr     │   │ _message │                       │
│  └──────────┘   └──────────┘                       │
│         │              │                             │
│         └──────────────┘                             │
│              HTTP                                    │
└──────────────────────────────────────┼───────────────┘
                                       │
┌──────────────────────────────────────┼───────────────┐
│  External APIs                       │               │
│  ┌──────────┐   ┌──────────┐         │               │
│  │  GitHub  │   │  Slack   │◀────────┘               │
│  │  REST    │   │  REST    │                         │
│  └──────────┘   └──────────┘                         │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env with your tokens
```

### 2. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 3. Run MCP Server

```bash
python -m mcp_server.main
```

### 4. Run LangGraph Agent

```bash
python -m langgraph.agent
```

### 5. Docker Compose (Full Stack)

```bash
docker-compose up -d
```

## Project Structure

```
multi-agent-supervisor/
├── mcp_server/          # MCP Server (Priority 1)
│   ├── server.py        # FastMCP entrypoint
│   ├── main.py          # Transport runner
│   └── tools/
│       ├── github.py    # github_create_pr
│       └── slack.py     # slack_send_message
├── langgraph/            # Orchestration Layer (Priority 2)
│   ├── state_machine.py  # AgentState + workflow graph
│   └── agent.py         # Supervisor agent entrypoint
├── docker-compose.yml    # Deployment scaffolding (Priority 3)
├── pyproject.toml
└── .env.example
```

## Tech Stack

- **LangGraph**: Agent workflow orchestration & state management
- **MCP Python SDK**: `mcp>=1.0.0` with `FastMCP` for stdio transport
- **LiteLLM**: Unified LLM proxy for OpenAI, Anthropic, local models
- **Pydantic**: Input/output validation and JSON Schema generation
- **httpx**: Async HTTP client for GitHub & Slack APIs
- **PostgreSQL + pgvector**: Vector store for agent memory
- **Docker Compose**: Multi-service orchestration

## License

MIT
