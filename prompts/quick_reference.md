# Cognitive Workspace Quick Reference

## The Four Cognitive Primitives

### 1. deconstruct
**Break problems into components**
```python
deconstruct(
    problem="Your complex problem here",
    max_depth=3  # Default: 3
)
```
Returns: root_id, component_ids[], tree

### 2. hypothesize  
**Find connections between concepts**
```python
hypothesize(
    node_a_id="thought_123",
    node_b_id="thought_456",
    context="Optional context"  # Recommended
)
```
Returns: hypothesis_id, hypothesis, similarity, path_count

### 3. synthesize
**Merge multiple concepts**
```python
synthesize(
    node_ids=["thought_1", "thought_2", "thought_3"],
    goal="What you want to achieve"  # Recommended
)
```
Returns: synthesis_id, synthesis, source_count

### 4. constrain
**Validate against rules**
```python
constrain(
    node_id="thought_999",
    rules=[
        "Must be technically feasible",
        "Must improve user experience",
        "Must be cost-effective"
    ]
)
```
Returns: overall_score, all_satisfied, rule_results[]

---

## Typical Workflow

```
1. DECONSTRUCT complex problem
   ↓ (get component_ids)
   
2. HYPOTHESIZE between components
   ↓ (explore connections)
   
3. SYNTHESIZE promising components
   ↓ (create unified insight)
   
4. CONSTRAIN synthesis
   ↓ (validate against requirements)
   
5. ✓ Solution validated!
```

---

## When to Use Each Tool

| Tool | Use When... | Don't Use When... |
|------|-------------|-------------------|
| **deconstruct** | Problem is complex, multi-faceted | Question is simple/direct |
| **hypothesize** | Exploring connections, finding synergies | Relationship is obvious |
| **synthesize** | Merging 2-5 related concepts | Concepts are unrelated |
| **constrain** | Validating reasoning | Rules are vague/contradictory |

---

## Score Interpretation

### Similarity (hypothesize)
- `> 0.7` = Strong connection
- `0.5-0.7` = Moderate connection
- `< 0.5` = Weak connection

### Constraints (constrain)
- `> 0.5` = Rule satisfied ✓
- `0.4-0.5` = Borderline (review)
- `< 0.4` = Rule not satisfied ✗

---

## Quick Examples

### Example 1: Simple Problem Solving
```
Problem: "Design a better keyboard"

1. deconstruct("Design ergonomic keyboard")
   → Components: layout, materials, feedback, connectivity, price

2. synthesize([layout, materials, feedback])
   → Insight: "Split design with mechanical switches and wrist support"

3. constrain(synthesis, ["Must cost < $150", "Must reduce RSI"])
   → Validated! ✓
```

### Example 2: Complex Research
```
Problem: "Improve ML model interpretability"

1. deconstruct("Improve ML model interpretability")
   → Components: feature importance, visualization, local explanations,
                 global structure, user interface

2. hypothesize(feature_importance, visualization)
   → "Interactive SHAP plots showing feature contributions"

3. hypothesize(local_explanations, global_structure)
   → "Hierarchical explanation system: instance → cluster → model"

4. synthesize([all hypotheses])
   → "Multi-level interactive explanation framework"

5. constrain(synthesis, ["Must work with any model", "< 100ms latency"])
   → Refine and iterate until constraints satisfied
```

---

## Tips

✓ Always provide context to hypothesize
✓ Give clear goals to synthesize  
✓ Use specific, measurable rules for constrain
✓ Start with deconstruct for complex problems
✓ Iterate if constraints aren't satisfied

✗ Don't use for simple questions
✗ Don't synthesize unrelated concepts
✗ Don't skip constraint validation
✗ Don't use vague rules

---

## Performance

- **deconstruct**: ~5-8s
- **hypothesize**: ~3-5s  
- **synthesize**: ~4-6s
- **constrain**: ~2-3s per rule

Total reasoning cycle: **~10-20 seconds**

---

## Node IDs

Format: `thought_<timestamp>`

Example: `thought_1763029351228711`

- Persist across sessions
- Stored in both Neo4j and Chroma
- Return from every tool call
- Use for subsequent operations

---

## Configuration

Edit `.env` file:
```bash
NEO4J_PASSWORD=your_password      # Required
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fast (384-dim)
# Or: Qwen/Qwen3-Embedding-0.6B   # Slow but better (1536-dim)
LLM_MODEL=qwen3:4b                # Recommended
CONFIDENCE_THRESHOLD=0.5          # Default
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| "Node not found" | Check node_id spelling |
| Low similarity | Concepts unrelated, revise |
| Constraints fail | Adjust synthesis or rules |
| Slow performance | Use all-MiniLM instead of Qwen |

---

For detailed explanations, see `tool_usage_guide.md`
