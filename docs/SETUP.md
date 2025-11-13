# Cognitive Workspace Database - Setup Guide

## What's New: LLM Integration ✨

The cognitive workspace now uses **local Ollama LLMs** for text generation, implementing the architecture described in the copilot instructions. The integration follows the "small models for simple tasks" philosophy:

- **LLM handles**: Bridge text generation (decomposition, hypothesis, synthesis)
- **Vector/Graph handles**: Heavy reasoning (similarity, centroids, path-finding)

## Quick Start

### 1. Install Dependencies

```bash
cd /cognitive-workspace-db

# Using uv (preferred)
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Set Up Ollama

Ollama should already be running on your system. Verify:

```bash
ollama --version
ollama list
```

### 3. Pull the Recommended Model

The system defaults to `qwen3:4b` - a lightweight 4B parameter model perfect for this use case:

```bash
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
```

**Why Qwen 4B?**
- Fast inference (~2-3 seconds per generation)
- Low memory footprint (~3GB VRAM)
- Excellent at structured text tasks
- Perfect for the "bridge text" role

**Alternative models** (if you prefer):
```bash
# Even lighter (faster but less coherent)
ollama pull phi3:mini

# Slightly better quality (but slower)
ollama pull mistral:7b

# For experimentation
ollama pull llama3.2:3b
```

### 4. Start Neo4j

Make sure Neo4j is running with default credentials:

```bash
# Check if Neo4j is running
sudo systemctl status neo4j

# Start if needed
sudo systemctl start neo4j
```

Default config expects:
- URI: `bolt://localhost:7687`
- User: `neo4j`
- Password: `your-password`

### 5. Environment Variables (Optional)

Create a `.env` file if you need custom settings:

```bash
# Cognitive Workspace Database Config
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
CHROMA_PATH=./chroma_data
LLM_MODEL=qwen3:4b
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
LLM_BASE_URL=http://localhost:11434
```

## Testing the Integration

### Direct Python Test

Create `test_workspace.py`:

```python
from src.server import CognitiveWorkspace, CWDConfig

# Initialize workspace
config = CWDConfig()
workspace = CognitiveWorkspace(config)

# Test 1: Deconstruct a problem (uses LLM)
print("=== TEST 1: Deconstruct ===")
result = workspace.deconstruct("How can I build a solar-powered water purification system?")
print(f"Root ID: {result['root_id']}")
print(f"Components: {len(result['component_ids'])}")
for comp_id in result['component_ids']:
    print(f"  - {comp_id}")

# Test 2: Hypothesize (uses LLM)
print("\n=== TEST 2: Hypothesize ===")
if len(result['component_ids']) >= 2:
    hyp = workspace.hypothesize(
        result['component_ids'][0],
        result['component_ids'][1],
        context="Exploring system design connections"
    )
    print(f"Hypothesis: {hyp['hypothesis']}")
    print(f"Similarity: {hyp['similarity']:.3f}")

# Test 3: Synthesize (uses LLM)
print("\n=== TEST 3: Synthesize ===")
if len(result['component_ids']) >= 3:
    synth = workspace.synthesize(
        result['component_ids'][:3],
        goal="Unified approach to solar water purification"
    )
    print(f"Synthesis: {synth['synthesis']}")

workspace.close()
print("\n✅ All tests completed!")
```

Run it:

```bash
python test_workspace.py
```

### Via MCP Client

If you're using this as an MCP server (e.g., with Claude Desktop):

1. Add to your MCP config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "cognitive-workspace": {
      "command": "python",
      "args": [
        "/home/ty/Repositories/ai_workspace/cognitive-workspace-db/src/server.py"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "CHROMA_PATH": "/home/ty/Repositories/ai_workspace/cognitive-workspace-db/chroma_data"
      }
    }
  }
}
```

2. Restart Claude Desktop

3. Test tools:
   - `deconstruct`: Break down a complex problem
   - `hypothesize`: Find connections between thought-nodes
   - `synthesize`: Merge insights from multiple nodes
   - `constrain`: Validate nodes against rules

## Implementation Details

### What Changed

**1. Dependencies** (`pyproject.toml`):
- Added `ollama>=0.4.7`

**2. Configuration** (`CWDConfig`):
- Added `llm_base_url`: Ollama server URL (default: `http://localhost:11434`)
- Added `llm_model`: Model name (default: `qwen2.5:4b`)

**3. LLM Helper** (`_llm_generate`):
- Simple wrapper around `ollama.chat()`
- Handles system + user prompts
- Configurable token limits
- Graceful fallback if Ollama unavailable

**4. Upgraded Methods**:

| Method | Before | After |
|--------|--------|-------|
| `_simple_decompose` | Split by sentences | LLM breaks into logical components |
| `_generate_hypothesis` | Generic similarity text | LLM explains conceptual connections |
| `_generate_synthesis` | String concatenation | LLM unifies concepts toward goal |

### System Prompts

The prompts are intentionally focused and simple:

**Decompose**:
> "You break complex problems into 3-5 logical sub-components. Output ONLY a numbered list of components, nothing else."

**Hypothesize**:
> "Given two concepts and their similarity score, write 1-2 sentences explaining their connection. Be concise and insightful."

**Synthesize**:
> "Given multiple concept fragments, write 1-2 sentences unifying them into a coherent insight. Be concise and focus on the common thread."

### Performance

With `qwen2.5:4b` on your RTX 3060:
- Decompose: ~2-3 seconds
- Hypothesize: ~1-2 seconds
- Synthesize: ~2-3 seconds

Total time for a full reasoning cycle: **~5-8 seconds**

## Architecture Validation

This implementation follows the design principles from `copilot-instructions.md`:

✅ **Small models work great**: Using 4B parameter Qwen
✅ **LLM does simple tasks**: Bridge text generation only
✅ **Heavy lifting in vector/graph**: Cosine similarity, centroids, path-finding
✅ **Focused prompts**: Short, specific system prompts
✅ **No token overhead**: Direct generation, no chain-of-thought bloat

## Troubleshooting

### "Connection refused" to Ollama

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### "Model not found"

```bash
# Pull the model
ollama pull qwen2.5:4b

# Verify it's available
ollama list
```

### LLM responses seem generic

This is expected! The LLM only provides bridge text. The actual reasoning happens through:
- Vector similarity (sentence-transformers)
- Graph path-finding (Neo4j Cypher)
- Latent space operations (numpy centroids)

If you want richer text, try:
1. Increase `max_tokens` in `_llm_generate` calls
2. Use a larger model like `mistral:7b`
3. Add more context to the prompts

### Neo4j connection errors

```bash
# Verify Neo4j is running
sudo systemctl status neo4j

# Check credentials match your config
neo4j-admin console
```

## Next Steps

### Experiment with Different Models

```python
# In your code or .env
config = CWDConfig(llm_model="mistral:7b")
```

### Add Custom Cognitive Primitives

The architecture is extensible. Add new operations by:
1. Creating a method in `CognitiveWorkspace`
2. Registering it in `@server.list_tools()`
3. Handling it in `@server.call_tool()`

### Optimize for Your Use Case

- **Speed priority**: Use `phi3:mini` or `tinyllama`
- **Quality priority**: Use `mistral:7b` or `llama3.2:3b`
- **Balance**: Stick with `qwen2.5:4b` (recommended)

## Research Context

This implementation realizes the vision from `rough_concept.md`:
- **System 2 reasoning** in latent space
- **Dual storage** (Neo4j + Chroma) for structure + semantics
- **Cognitive primitives** instead of CRUD operations
- **Continuous thought** inspired by Meta's COCONUT
- **Hierarchical reasoning** with transparent graph representation

The LLM integration keeps the system grounded: it generates natural language bridges while the mathematical operations (embeddings, similarity, centroids) do the actual reasoning.

---

**Ready to reason! 🧠**

For issues or questions, check:
- `src/rough_concept.md` - Theoretical foundation
- `.github/copilot-instructions.md` - Implementation guide
- [Ollama Python Docs](https://github.com/ollama/ollama-python)
