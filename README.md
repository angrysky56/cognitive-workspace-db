# Cognitive Workspace Database

> **A "System 2" reasoning engine that operates in latent space**

Hybrid Neo4j + Chroma architecture for AI agents to perform deliberate, multi-step reasoning using cognitive primitives instead of traditional CRUD operations.

## What Makes This Different

Traditional memory databases store and retrieve. This workspace **thinks**:

- **Deconstruct**: Break complex problems into hierarchical thought-nodes
- **Hypothesize**: Discover novel connections through graph paths + vector similarity
- **Synthesize**: Merge concepts by computing latent space centroids
- **Constrain**: Validate reasoning by projecting thoughts against rule-vectors

## Architecture

```
┌─────────────────────────────────────────────┐
│  Cognitive Primitives (MCP Server)          │
│  ├─ deconstruct()   ├─ synthesize()         │
│  ├─ hypothesize()   └─ constrain()          │
└─────────────────────────────────────────────┘
           ↓                    ↓
    ┌──────────────┐     ┌──────────────┐
    │   Neo4j      │     │   Chroma     │
    │  (Structure) │     │  (Vectors)   │
    └──────────────┘     └──────────────┘
         Graph                Embeddings
    Path-finding         Cosine similarity
    Relationships         Centroids
```

**+ Local LLM (Ollama)**: Generates bridge text for decomposition, hypothesis, and synthesis

## Quick Start

```bash
# 1. Install dependencies
cd /cognitive-workspace-db
uv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
uv sync

# uv pip install -e .

# 2. Configure environment
cp .env.example .env
# Edit .env and set NEO4J_PASSWORD

# 3. Pull the LLM models (suggested)
ollama pull qwen3:4b                    # For text generation
# Optional: Use Qwen embeddings for better quality (1.5GB download)
# Default is all-MiniLM-L6-v2 (80MB, fast, good quality)
# ollama pull qwen3-embedding:0.6b  # Not needed, sentence-transformers handles it

Add to your AI Client for MCP Server:

```json
{
  "mcpServers": {
    "cognitive-workspace-db": {
      "command": "uv",
      "args": [
        "--directory",
        "path-to-your/cognitive-workspace-db/src",
        "run",
        "server.py"
      ]
    }
  }
}
```

# 4. Ensure Neo4j is running ie start your neo4j Desktop and db.
sudo systemctl start neo4j

# 5. Test it
python test_workspace.py
```

## Example: Reasoning in Action

```python
from src.server import CognitiveWorkspace, CWDConfig

workspace = CognitiveWorkspace(CWDConfig())

# Break down a complex problem
result = workspace.deconstruct(
    "How can I build an autonomous greenhouse with AI?"
)

# Find connections between components
hypothesis = workspace.hypothesize(
    result['component_ids'][0],
    result['component_ids'][1],
    context="Exploring synergies"
)

# Merge insights
synthesis = workspace.synthesize(
    result['component_ids'][:3],
    goal="Integrated automation approach"
)

# Validate against constraints
validation = workspace.constrain(
    synthesis['synthesis_id'],
    rules=["Must be feasible", "Must be cost-effective"]
)
```

## Research Foundation

Implements cutting-edge concepts:
- **Meta's COCONUT** (Dec 2024): Continuous thought in latent space
- **Hierarchical Reasoning Models**: Direct latent transformations
- **Knowledge Graph Reasoning**: Structural path-finding for inference

**Goal**: Move AI from "System 1" (pattern matching) → "System 2" (deliberate reasoning)

## Why Small Models Work

The LLM does **simple bridge text** generation:
- "Break this into 3-5 parts"
- "Explain this connection in 1 sentence"
- "Unify these concepts"

The **heavy reasoning** happens in vector/graph operations:
- Cosine similarity (numpy)
- Embedding centroids (sentence-transformers)
- Path-finding (Neo4j Cypher)

**Result**: A 4B model like Qwen handles this perfectly. No need for 70B+ models.

## Performance

With `qwen3:4b` (LLM) + `all-MiniLM-L6-v2` (embeddings):
- Full reasoning cycle: **~5-8 seconds** (CPU)
- Memory footprint: **~3GB** (CPU mode)
- Cost: **$0** (fully local)

With `Qwen/Qwen3-Embedding-0.6B` (optional):
- Better embedding quality, ~2x slower
- GPU recommended (automatically uses flash_attention_2 if available)

## Use as MCP Server

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "cognitive-workspace": {
      "command": "python",
      "args": [
        "/home/ty/Repositories/ai_workspace/cognitive-workspace-db/src/server.py"
      ]
    }
  }
}
```

Claude can then use cognitive primitives as tools for complex reasoning tasks.

## Documentation

- **[SETUP.md](./SETUP.md)**: Complete setup guide, troubleshooting, architecture details
- **[copilot-instructions.md](./.github/copilot-instructions.md)**: Implementation patterns and conventions
- **[rough_concept.md](./src/rough_concept.md)**: Theoretical foundation

## Current State

✅ Four cognitive primitives implemented
✅ LLM integration (Ollama) complete
✅ Dual storage (Neo4j + Chroma) synchronized
✅ MCP server ready
⏳ Tests (coming soon)

## Tech Stack

- **Python 3.12+** with modern type hints
- **Neo4j 5.27+** for graph operations
- **ChromaDB 0.5+** for vector storage
- **sentence-transformers** for embeddings
  - Default: `all-MiniLM-L6-v2` (384-dim, 80MB, fast)
  - Optional: `Qwen/Qwen3-Embedding-0.6B` (1536-dim, 1.5GB, higher quality)
- **Ollama** for local LLM (qwen3:4b recommended)
- **MCP SDK 1.1+** for tool integration

## License

[Add your license here]

## Acknowledgments

Inspired by:
- Meta AI Research: COCONUT paper on continuous thought
- Knowledge graph reasoning literature
- Hierarchical reasoning model research
- The quest for System 2 reasoning in AI

---

**Ready to think beyond token generation.** 🧠
