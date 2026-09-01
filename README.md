# Agentic-OS

An agentic research operating system that routes tasks to specialized workers, scrapes and synthesizes information from the web, and persists every run in a local SQLite registry — all with a free-first provider strategy that works without paid API keys.

## How It Works

```
User Objective
      │
      ▼
  Manager (routes by intent)
      │
      ├── researcher ──► normal search + extract + synthesize
      │                   └── deep research (comprehensive queries)
      │
      └── browser_worker ──► open & inspect a URL in a real browser
```

The **Manager** analyses each objective, selects a worker and tool, records the run in SQLite, executes it, and writes back the result — including status, duration, and any error.

## Features

- **Free-first provider routing** — Jina extraction works with zero API keys; Brave / Tavily / Exa are used for search when configured
- **Reserve-only Firecrawl** — never called unless explicitly allowed, protecting your quota
- **Adaptive routing** — providers that fail are deprioritised automatically (`AGENT_OS_ADAPTIVE_ROUTING=true`)
- **LLM synthesis** — optional Gemini-powered answer generation on top of scraped sources
- **Deep research** — comprehensive multi-source reports triggered by keywords like "compare", "in-depth", "all major"
- **Run registry** — every job logged to SQLite with retry support and worker leases
- **REST API** — FastAPI control plane at `/v1/jobs`, `/v1/runs`, `/v1/runtime`
- **MCP server** — expose the agent as an MCP tool for Claude Desktop
- **235 tests** — full pytest suite covering routing, provider fallback, retry logic, and synthesis

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install

```bash
git clone https://github.com/thecrypticcontroller/Agentic-OS.git
cd Agentic-OS
uv sync
```

### Configure

Copy `.env.example` to `.env` and fill in what you have (everything is optional):

```bash
cp .env.example .env
```

```env
# Search providers (at least one recommended)
BRAVE_API_KEY=         # https://api.search.brave.com/register  (2,000/mo free)
TAVILY_API_KEY=        # https://app.tavily.com                 (1,000/mo free)
EXA_API_KEY=           # https://dashboard.exa.ai              (1,000/mo free)

# LLM synthesis (optional — raw findings returned if omitted)
GEMINI_API_KEY=        # https://aistudio.google.com/apikey    (free)
AGENT_OS_LLM_ENABLED=true

# Reserve provider (only used when explicitly enabled)
FIRECRAWL_API_KEY=     # https://firecrawl.dev

# Feature flags
AGENT_OS_ADAPTIVE_ROUTING=true
```

> **Zero-key mode:** Jina extraction works with no API keys at all.  
> URL-based jobs (`Research https://...`) complete out of the box.

### Run a research job

```python
from agents.manager import Manager
import json

m = Manager()
job = m.create_job("Research Python FastAPI best practices")
result = m.execute(job)

data = json.loads(result.result)
print("Answer:", data["answer"])
for f in data["key_findings"]:
    print("-", f)
```

### Run the API server

```bash
uv run uvicorn api.app:app --reload
```

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/jobs` | Submit a new job |
| `GET` | `/v1/runs` | List all runs |
| `GET` | `/v1/runs/{id}` | Get a specific run |
| `POST` | `/v1/runs/{id}/retry` | Retry a failed run |
| `GET` | `/v1/runtime` | Provider and capacity status |

### Run as MCP server (Claude Desktop)

```bash
uv run python mcp_server.py
```

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "agent-os": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/path/to/Agentic-OS"
    }
  }
}
```

### Run tests

```bash
uv run pytest -q
# 235 passed
```

## Project Structure

```
Agentic-OS/
├── agents/
│   ├── manager.py        # Orchestrator: routing, planning, execution
│   ├── researcher.py     # Web search + scrape + deep research
│   ├── synthesizer.py    # LLM-powered answer generation
│   └── browser_worker.py # Browser-based page inspection
├── tools/
│   ├── web_research.py   # Jina / Firecrawl / search adapters
│   ├── provider_router.py# Free-first provider selection
│   ├── run_registry.py   # SQLite run persistence
│   ├── retry_policy.py   # Retry logic with error classification
│   └── deep_research.py  # Comprehensive multi-source research
├── api/
│   └── app.py            # FastAPI control plane
├── workers/              # Queue worker (concurrent job processing)
├── tests/                # 235 pytest tests
├── mcp_server.py         # MCP server interface
└── main.py               # CLI demo
```

## Provider Strategy

| Provider | Role | Requires Key |
|----------|------|-------------|
| Jina | Web extraction | No key needed |
| Brave | Web search | Free tier |
| Tavily | Web search | Free tier |
| Exa | Search + extract | Free tier |
| Firecrawl | Reserve only | Paid |
| Gemini | LLM synthesis | Free tier |

Providers are tried in priority order. Firecrawl is never called unless `allow_reserve=True` is passed explicitly.

## Tech Stack

- **Python 3.11+** with uv
- **FastAPI** — REST control plane
- **SQLite** — run registry and worker leases
- **Jina / Tavily / Brave / Exa** — web search and extraction
- **Google Gemini** — LLM synthesis
- **MCP** — Claude Desktop integration
- **pytest** — 235-test suite

## License

MIT
