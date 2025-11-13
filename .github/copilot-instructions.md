# Cognitive Workspace Database - AI Agent Instructions

## Project Vision

This is a **"System 2" reasoning engine** - not a traditional memory database, but an **active cognitive workspace** for AI agents. Think "workbench" not "filing cabinet". Implements cutting-edge research concepts (Meta's COCONUT, hierarchical reasoning models) to enable multi-step reasoning in latent space.

## Architecture: Hybrid Reasoning System

```
Neo4j (Graph) + Chroma (Vectors) = Cognitive Primitives
```

- **Neo4j**: Structural reasoning through knowledge graphs. Stores `ThoughtNode` entities with typed relationships (`DECOMPOSES_INTO`, `SYNTHESIZES_FROM`, `CONNECTS`, `SUPPORTS`)
- **Chroma**: Latent space operations. Embeddings generated via `sentence-transformers` (default: `all-MiniLM-L6-v2`)
- **MCP Server**: Exposes cognitive operations as tools, not CRUD endpoints

## Four Cognitive Primitives (Not CRUD Operations)

1. **`deconstruct(problem)`**: Break complex vectors into hierarchical thought-nodes. Creates queryable reasoning trees similar to breadth-first search in Meta's COCONUT
2. **`hypothesize(node_a, node_b)`**: Find novel connections using graph paths + vector similarity. Operates in conceptual space
3. **`synthesize(node_ids, goal)`**: Merge disparate thoughts by computing latent space centroids. Inspired by hierarchical reasoning models
4. **`constrain(node_id, rules)`**: Validate reasoning by projecting thought-vectors against rule-vectors. Enables "self-checking"

## Key Implementation Patterns

### Thought-Node Schema
Every thought is dual-stored:
- **Neo4j**: Metadata (id, content, cognitive_type, confidence, created_at, parent_problem)
- **Chroma**: Embedding vector + searchable metadata

### Cognitive Types (not data types)
- `problem`: Root decomposition target
- `sub_problem`: Decomposed component
- `hypothesis`: Discovered connection
- `synthesis`: Merged insight

### Vector Operations
All cognitive functions operate through `_embed_text()` → numpy operations → graph relationships. Never bypass this flow.

**Embedding modes**:
- `_embed_text(text)`: Document embedding (for storage)
- `_embed_text(text, is_query=True)`: Query embedding (for similarity search with Qwen models)

Qwen models automatically use `prompt_name="query"` for better retrieval when `is_query=True`.

## Configuration (Environment-Driven)

**NEVER hardcode credentials in `server.py`!** Use `.env` file or environment variables.

`.env` file should be in the **project root**, not in `src/`. The config automatically searches:
1. Current directory
2. Parent directory (project root when MCP server runs from `src/`)

```bash
# Copy .env.example to .env and configure
cp .env.example .env

# Required in .env:
NEO4J_PASSWORD=your_secure_password

# Optional (have defaults):
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
CHROMA_PATH=./chroma_data
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=qwen2.5:4b
```

Config loads automatically via `pydantic-settings`.

**Embedding models**:
- Default: `all-MiniLM-L6-v2` (384-dim, fast, 80MB)
- Qwen: `Qwen/Qwen3-Embedding-0.6B` (1536-dim, higher quality)
  - CPU: Uses standard attention with left padding
  - GPU: Attempts flash_attention_2 if available (requires `pip install flash-attn`), falls back gracefully

## Development Workflow

### Running the MCP Server
```bash
# Install dependencies
pip install -e .

# Run server (stdio mode for MCP clients)
python src/server.py
```

The server uses **stdio transport** - it's designed to be invoked by MCP hosts, not run as a standalone HTTP service.

### Testing Cognitive Primitives
Use an MCP client or test via direct method calls to `CognitiveWorkspace` class. No REST endpoints.

### Database Initialization
Neo4j and Chroma initialize on first use. No migrations needed - schema is code-defined.

### Example Workflow
```python
# Direct testing (without MCP client)
from src.server import CognitiveWorkspace, CWDConfig

workspace = CognitiveWorkspace(CWDConfig())

# 1. Deconstruct a problem
with workspace.neo4j_driver.session() as session:
    result = workspace.deconstruct("How do I build a treehouse?")
    root_id = result["root_id"]
    component_ids = result["component_ids"]

# 2. Hypothesize connections
with workspace.neo4j_driver.session() as session:
    if len(component_ids) >= 2:
        hyp = workspace.hypothesize(component_ids[0], component_ids[1])
        print(f"Hypothesis: {hyp['hypothesis']}")

# 3. Synthesize insights
with workspace.neo4j_driver.session() as session:
    synthesis = workspace.synthesize(component_ids[:3], goal="Building approach")
    print(f"Synthesis: {synthesis['synthesis']}")
```

### Testing Strategy (Future)
When adding tests, focus on:
- **Dual-storage integrity**: Verify nodes exist in both Neo4j and Chroma with matching IDs
- **Vector operations**: Check cosine similarity calculations, centroid computation
- **Graph traversal**: Validate relationship creation and path-finding
- **Idempotency**: Same input should produce consistent embeddings

## Critical Dependencies

- **mcp**: Model Context Protocol SDK (>= 1.1.2)
- **neo4j**: Graph database driver (>= 5.27.0)
- **chromadb**: Vector store (>= 0.5.23)
- **sentence-transformers**: Embedding generation (>= 3.3.1)
- **accelerate**: Required for flash_attention_2 optimization with Qwen embeddings (>= 1.2.1)

## Code Conventions

- Use Python 3.12+ features (`from __future__ import annotations`, type hints)
- Pydantic for config models (see `CWDConfig`)
- Async MCP handlers (`@server.list_tools()`, `@server.call_tool()`)
- Log extensively via `logging.getLogger("cwd-mcp")`

## LLM Integration (Lightweight Models Work Great)

The cognitive primitives need **simple text generation**, not deep reasoning. Heavy lifting happens in vector/graph operations. A **4B model like Qwen via Ollama is sufficient**.

**Note**: LLM output is automatically cleaned (strips `<think>` tags from Qwen models).

### What LLMs Do (Simple Tasks)
- `_simple_decompose`: Break problem text into 3-5 logical parts
- `_generate_hypothesis`: Write 1-2 sentence connection between concepts (given similarity score)
- `_generate_synthesis`: Merge concept previews into unified statement

### What Vector/Graph Do (Heavy Lifting)
- Cosine similarity, embedding centroids (numpy)
- Path-finding, relationship traversal (Neo4j Cypher)
- Semantic search (Chroma + sentence-transformers)

### Example Integration Pattern
```python
# Add to CWDConfig
llm_base_url: str = Field(default="http://localhost:11434")  # Ollama
llm_model: str = Field(default="qwen2.5:4b")

# Simple helper in CognitiveWorkspace
def _llm_generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 100) -> str:
    response = requests.post(f"{self.config.llm_base_url}/api/generate", json={
        "model": self.config.llm_model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    })
    return response.json()["response"]
```

### System Prompts (Keep Them Focused)
```python
# For decompose
"You break problems into 3-5 logical sub-components. Output only the list."

# For hypothesize
"Given two concepts and similarity={score}, write 1 sentence explaining their connection."

# For synthesize
"Given {n} concept fragments, write 1 sentence unifying them toward: {goal}"
```

**Why small models work**: They're generating bridge text, not solving problems. The cognitive workspace does the reasoning through vector/graph operations.

## Critical Patterns to Maintain

### Dual-Storage Synchronization
`_create_thought_node()` is the only entry point for creating nodes - always writes to both Neo4j AND Chroma atomically. Never create nodes directly:
```python
# ❌ WRONG - breaks sync
session.run("CREATE (t:ThoughtNode {...})")
self.collection.add(...)

# ✅ CORRECT - use the helper
self._create_thought_node(session, content, cognitive_type, ...)
```

### Session Management Pattern
Neo4j operations use context managers - never store sessions as instance variables:
```python
with self.neo4j_driver.session() as session:
    # All Cypher queries here
    result = session.run("MATCH ...")
```

### Adding New Relationship Types
When extending the graph schema, update both the Cypher queries AND the documentation:
1. Add relationship in cognitive primitive: `CREATE (a)-[:NEW_TYPE {weight: $w}]->(b)`
2. Document in "Key Implementation Patterns" section above
3. Example types: `DECOMPOSES_INTO`, `SYNTHESIZES_FROM`, `CONNECTS`, `SUPPORTS`

## Debugging Tips

### Inspect the Graph
```bash
# Neo4j Browser: http://localhost:7474
# Query all thought nodes:
MATCH (t:ThoughtNode) RETURN t LIMIT 25

# Query relationships:
MATCH (a)-[r]->(b) RETURN a.content, type(r), b.content LIMIT 50
```

### Check Embeddings
```python
# In Python REPL with workspace loaded:
from src.server import CognitiveWorkspace, CWDConfig
workspace = CognitiveWorkspace(CWDConfig())

# Check embedding dimension
test_vec = workspace._embed_text("test")
print(f"Dimension: {len(test_vec)}")  # Should be 384 for all-MiniLM-L6-v2

# Query Chroma
results = workspace.collection.get(ids=["thought_12345"])
```

### Logging
All cognitive operations log via `logger.info()` - check stdout for MCP server output.

## What NOT to Do

- Don't add traditional CRUD endpoints - this isn't a REST API
- Don't bypass dual-storage (Neo4j + Chroma must stay in sync via `_create_thought_node`)
- Don't introduce token-based reasoning - stay in latent space
- Don't create generic memory tools - focus on cognitive operations
- Don't modify Neo4j nodes without updating Chroma embeddings

## Current State (MVP)

Single MCP server (`src/server.py`) with 4 cognitive primitives. Simplified logic for hypothesis/synthesis generation (production requires LLM integration). No tests yet.

**Design document**: `src/rough_concept.md` explains the theoretical foundation.

## Research Context

This implements concepts from:
- Meta's COCONUT (Dec 2024): Latent-space reasoning via continuous thought
- Hierarchical Reasoning Models: Direct latent transformations without token overhead
- Knowledge Graph Reasoning: Structural path-finding for logical inference

Goal: Move AI from "System 1" (pattern matching) to "System 2" (deliberate reasoning).
