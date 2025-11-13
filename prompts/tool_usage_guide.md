# Cognitive Workspace Tools - AI Usage Guide

This guide explains how to effectively use the cognitive-workspace-db MCP server tools for System 2 reasoning tasks.

## Overview

The Cognitive Workspace is a "System 2" reasoning engine that operates in latent space, combining:
- **Neo4j** for structural/graph reasoning
- **Chroma** for vector operations
- **Ollama LLM** for bridge text generation

It provides four cognitive primitives that enable deliberate, multi-step reasoning.

---

## Tool 1: deconstruct

**Purpose**: Break complex problems into hierarchical component thought-nodes

**When to Use**:
- User asks to break down a complex problem
- Need to understand problem structure before solving
- Want to create a reasoning tree for exploration

**Best Practices**:
```
✓ Use for genuinely complex problems that benefit from decomposition
✓ Start with max_depth=3 for most tasks
✓ The problem should be substantial enough to decompose meaningfully
✗ Don't use for simple questions that don't need breakdown
✗ Avoid trivial decompositions (e.g., "What is 2+2?")
```

**Example Usage**:
```
Problem: "How can we build a sustainable urban transportation system?"

Result: Creates tree with root problem and 3-5 sub-problems like:
- Infrastructure requirements
- Energy sources and emissions
- Public adoption strategies
- Economic viability
- Integration with existing systems
```

**What You Get**:
- `root_id`: ID of the root problem node
- `component_ids`: List of sub-problem node IDs
- `tree`: Hierarchical structure showing decomposition

**Next Steps After Deconstructing**:
1. Use `hypothesize` to find connections between components
2. Use `synthesize` to merge related components
3. Use `constrain` to validate components against requirements

---

## Tool 2: hypothesize

**Purpose**: Discover novel connections between two concepts using graph paths + vector similarity

**When to Use**:
- Need to find relationships between decomposed components
- Exploring potential synergies or conflicts
- Looking for creative connections in reasoning

**Best Practices**:
```
✓ Use after deconstruct to connect components
✓ Provide meaningful context to guide hypothesis generation
✓ Connect concepts that might have non-obvious relationships
✗ Don't connect nodes that are obviously related (waste of computation)
✗ Avoid when direct logical relationship is already clear
```

**Example Usage**:
```
node_a_id: "thought_123" (Infrastructure requirements)
node_b_id: "thought_456" (Public adoption strategies)
context: "Exploring how infrastructure design affects user adoption"

Result: Hypothesis about how visible, accessible infrastructure increases public trust and adoption
```

**What You Get**:
- `hypothesis_id`: ID of the generated hypothesis node
- `hypothesis`: Text explaining the connection
- `similarity`: Cosine similarity score (0-1)
- `path_count`: Number of graph paths found

**Interpreting Results**:
- Similarity > 0.7: Strong semantic connection
- Similarity 0.5-0.7: Moderate connection worth exploring
- Similarity < 0.5: Weak connection, may not be useful
- path_count > 0: Structural relationship exists in graph

---

## Tool 3: synthesize

**Purpose**: Merge multiple thought-nodes into unified insight by computing latent space centroids

**When to Use**:
- Need to combine multiple concepts into coherent whole
- Looking for common patterns across ideas
- Want to create higher-level abstractions

**Best Practices**:
```
✓ Synthesize 2-5 related concepts for best results
✓ Provide a clear goal to guide synthesis
✓ Use after hypothesize to merge connected components
✗ Don't synthesize unrelated concepts (produces gibberish)
✗ Avoid synthesizing too many nodes (> 7) - becomes unfocused
```

**Example Usage**:
```
node_ids: ["thought_123", "thought_456", "thought_789"]
goal: "Create integrated approach for sustainable transportation"

Result: Unified insight combining infrastructure, adoption, and economics into coherent strategy
```

**What You Get**:
- `synthesis_id`: ID of the synthesis node
- `synthesis`: Text of the unified insight
- `source_count`: Number of nodes synthesized

**The Math Behind It**:
- Computes centroid (geometric center) of input node embeddings
- Creates synthesis node positioned at centroid in latent space
- This represents the "average" or "intersection" of input concepts

---

## Tool 4: constrain

**Purpose**: Validate thought-nodes against rules by projecting in latent space

**When to Use**:
- Need to validate reasoning against requirements
- Checking if solution meets constraints
- Verifying hypothesis or synthesis is sound

**Best Practices**:
```
✓ Use clear, specific rules
✓ Validate synthesis/hypothesis nodes after creation
✓ Apply 3-5 constraints for comprehensive validation
✗ Don't use vague rules like "must be good"
✗ Avoid contradictory rules
```

**Example Usage**:
```
node_id: "thought_synthesis_999"
rules: [
  "Must be technically feasible",
  "Must improve user experience", 
  "Must enhance system reliability",
  "Must be cost-effective"
]

Result: Scores for each rule + overall satisfaction
```

**What You Get**:
- `overall_score`: Average score across all rules (0-1)
- `all_satisfied`: Boolean - do all rules pass threshold?
- `rule_results`: Individual scores for each rule

**Interpreting Scores**:
- Score > 0.5 (threshold): Rule is satisfied
- Score 0.4-0.5: Borderline, needs review
- Score < 0.4: Rule not satisfied, revise thought-node

---

## Complete Reasoning Workflow Example

**Problem**: "Design an AI-powered greenhouse automation system"

### Step 1: Deconstruct
```
deconstruct(
  problem="Design an AI-powered greenhouse automation system",
  max_depth=3
)

→ Components:
  - Climate control systems
  - Water and nutrient management
  - Pest detection and response
  - Energy optimization
  - Crop health monitoring
```

### Step 2: Hypothesize Connections
```
hypothesize(
  node_a_id="climate_control_node",
  node_b_id="energy_optimization_node",
  context="Exploring energy-efficient climate control"
)

→ Hypothesis: "Smart HVAC scheduling based on plant growth phases 
   reduces energy consumption while maintaining optimal conditions"
```

### Step 3: Synthesize Approach
```
synthesize(
  node_ids=["climate_control_node", "water_mgmt_node", "crop_monitoring_node"],
  goal="Integrated environmental management system"
)

→ Synthesis: "Unified IoT sensor network feeding ML models that 
   coordinate climate, water, and monitoring for optimized growing conditions"
```

### Step 4: Validate with Constraints
```
constrain(
  node_id="synthesis_node",
  rules=[
    "Must be cost-effective for small farms",
    "Must work with standard greenhouse equipment",
    "Must reduce water usage by >20%",
    "Must improve crop yield"
  ]
)

→ All satisfied: true (overall score: 0.72)
   System meets all requirements!
```

---

## Tips for Best Results

### 1. Problem Selection
- **Good**: Complex, multi-faceted problems that benefit from structured thinking
- **Bad**: Simple questions with direct answers

### 2. Tool Sequencing
- Always start with `deconstruct` for complex problems
- Use `hypothesize` to explore relationships between components
- Apply `synthesize` when you have 2-5 related concepts to merge
- Finish with `constrain` to validate your reasoning

### 3. Context Matters
- Provide specific context to `hypothesize` for better connections
- Give clear goals to `synthesize` for focused results
- Use precise rules for `constrain` validation

### 4. Iterative Refinement
- If synthesis doesn't satisfy constraints, revise approach
- Try different combinations of nodes for synthesis
- Explore multiple hypotheses before synthesizing

### 5. When NOT to Use These Tools
- Simple factual questions → Just answer directly
- Quick calculations → No reasoning tree needed
- Already have clear solution → Don't over-engineer

---

## Technical Notes

### Performance Expectations
- **Deconstruct**: ~5-8 seconds (LLM decomposition + graph creation)
- **Hypothesize**: ~3-5 seconds (vector similarity + graph search)
- **Synthesize**: ~4-6 seconds (centroid computation + LLM generation)
- **Constrain**: ~2-3 seconds per rule (vector projections)

### Model Requirements
- **LLM**: Ollama qwen3:4b (or similar 4B+ model)
- **Embeddings**: Qwen3-Embedding-0.6B or all-MiniLM-L6-v2
- **Databases**: Neo4j 5.27+ and ChromaDB 0.5+

### Node IDs
- Format: `thought_<timestamp_microseconds>`
- Unique across all sessions
- Persistent in Neo4j + Chroma
- Use returned IDs for subsequent operations

---

## Common Patterns

### Pattern 1: Problem Decomposition → Solution Synthesis
```
1. deconstruct(complex_problem)
2. for each component_pair:
     hypothesize(comp_a, comp_b, context)
3. synthesize(all_components, goal)
4. constrain(synthesis, requirements)
```

### Pattern 2: Iterative Refinement
```
1. deconstruct(problem)
2. synthesize(components[0:3])
3. constrain(synthesis, rules)
4. if not all_satisfied:
     - Revise components or try different synthesis
     - Repeat until constraints satisfied
```

### Pattern 3: Exploration Mode
```
1. deconstruct(problem)
2. for each component:
     for each other_component:
       hypothesize(component, other_component)
3. Review all hypotheses to find novel insights
4. synthesize most promising connections
```

---

## Troubleshooting

### Issue: "Node not found"
- **Cause**: Using node_id from different session or typo
- **Fix**: Use node IDs returned by tools in same session

### Issue: Low similarity scores
- **Cause**: Concepts are genuinely unrelated
- **Fix**: Choose more related nodes or revise problem decomposition

### Issue: Synthesis not satisfying constraints
- **Cause**: Components don't align with requirements
- **Fix**: Add more relevant components or adjust constraints

### Issue: LLM output is verbose
- **Cause**: Model generating reasoning artifacts
- **Fix**: This is filtered automatically, check output field only

---

## Advanced Usage

### Custom Confidence Thresholds
The default threshold is 0.5. For stricter validation:
- Set CONFIDENCE_THRESHOLD=0.6 or higher in .env
- Reduces false positives in constraint validation

### Graph Queries
Thought-nodes persist in Neo4j. You can query directly:
```cypher
// Find all decompositions
MATCH (p:ThoughtNode)-[:DECOMPOSES_INTO]->(c:ThoughtNode)
RETURN p, c

// Find all hypotheses with high similarity
MATCH (h:ThoughtNode {cognitive_type: 'hypothesis'})-[r:CONNECTS]->(n)
WHERE r.similarity > 0.7
RETURN h, n, r.similarity
```

### Embedding Dimensions
- all-MiniLM-L6-v2: 384 dimensions (fast, good quality)
- Qwen3-Embedding-0.6B: 1536 dimensions (slower, higher quality)

---

## Summary

The Cognitive Workspace enables **System 2 reasoning** through four operations:

1. **deconstruct**: Problem → Components
2. **hypothesize**: Component A + Component B → Connection
3. **synthesize**: Multiple Components → Unified Insight  
4. **constrain**: Insight + Rules → Validation

Use these tools for complex reasoning tasks that benefit from structured, deliberate thinking in latent space.
