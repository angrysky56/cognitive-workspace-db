# Gen 3 Utility-Guided Architecture

## Overview

The Cognitive Workspace Database (CWD) now implements a **Gen 3 Utility-Guided Architecture** that combines:

1. **Schmidhuber's Compression Progress** - Intrinsic curiosity rewards for learning
2. **Utility-Guided Exploration** - Directed learning focused on goal-aligned patterns
3. **Topology Tunneling** - Analogical reasoning for "Aha!" moments
4. **Knowledge Compression as Tools** - Converting solved problems into reusable patterns

This transforms the CWD from a passive reasoning database into an active, goal-driven learning system.

---

## Quick Start: New Tools

### Set Goals (Utility Director)
```python
goal_id = workspace.set_goal(
    "Learn automotive electrical diagnostics",
    utility_weight=1.0
)
```

### Compress Knowledge to Tools
```python
tool_id = workspace.compress_to_tool(
    node_ids=["thought_1", "thought_2"],
    tool_name="pry_plastic_housing_safely",
    description="General method for removing plastic housings"
)
```

### Explore High-Value Learning
```python
candidates = workspace.explore_for_utility(
    focus_area="electrical systems",
    max_candidates=10
)
# Returns nodes with high utility × compression_potential
```

### Enhanced Topology Tunneling
```python
# System finds analogical patterns from tool library
hypothesis = workspace.hypothesize(
    node_a_id="stuck_housing",
    node_b_id="jar_lid_concept"
)
# "Aha!" - uses jar lid technique for car housing
```

---

## Key Concepts

### The Three Generations

**Gen 1: Standard RL** - Maximize extrinsic reward → Stops exploring after first solution

**Gen 2: Compression Progress** - Add intrinsic curiosity → "Junk food" problem (learns useless patterns)

**Gen 3: Utility-Guided** - Filter curiosity through goals → Directed learning

### Core Equation
```
Total Reward = Extrinsic + (Utility Score × Compression Progress)
```

**Utility Score** = Vector similarity to active goals
**Compression Progress** = `old_compression - new_compression`

### Schmidhuber's Insight

**True novelty** exists in learnable regularities, not randomness:
- **Too Simple** (fully compressed) → Boring
- **Too Random** (incompressible) → Boring
- **Learnable Patterns** (compressible with effort) → **Interesting!**

---

## Example: Jeep Brake Light Problem

```python
# 1. Set utility goal
workspace.set_goal("Fix 2003 Jeep Liberty brake light")

# 2. Deconstruct (System 2)
result = workspace.deconstruct(
    "Replace rear brake light bulb on 2003 Jeep Liberty"
)
# Returns: ["Identify bulb", "Access housing", "Purchase bulb"]

# 3. Fast solutions (System 1)
# - "Purchase bulb" → Has tool: buy_auto_parts_online ✓
# - "Access housing" → Stuck! No fast solution ✗

# 4. Topology tunneling (analogical leap)
hypothesis = workspace.hypothesize(
    node_a_id="housing_stuck",
    node_b_id="jar_lid_concept"  # System finds this analogy
)
# Result: "Use flat tool with even pressure like jar lid removal"

# 5. Validate utility
workspace.constrain(
    node_id=hypothesis_id,
    rules=["Don't break plastic", "Use common tools"]
)
# Passes ✓

# 6. Compress to new tool
workspace.compress_to_tool(
    node_ids=[housing_id, bulb_id],
    tool_name="change_jeep_brake_light"
)
# Next time: instant System 1 solution
```

---

## Architecture Enhancements

### Database Schema

**ThoughtNode (Enhanced)**:
```cypher
(:ThoughtNode {
  // Existing fields
  id, content, cognitive_type, confidence,

  // Gen 3 additions
  utility_score: float,        // Goal alignment
  compression_score: float,    // Compressibility
  intrinsic_reward: float      // Learning reward
})
```

**New Node Types**:
```cypher
(:Goal {id, description, utility_weight, active})
(:Tool {id, name, pattern, usage_count, success_rate})
```

**New Relationships**:
```cypher
(Tool)-[:COMPRESSED_FROM]->(ThoughtNode)
(Hypothesis)-[:INSPIRED_BY]->(Tool)
```

### In-Memory Components

- `active_goals`: Current goal dictionary
- `compression_history`: Learning curves per node
- `tool_library`: Fast access to compressed knowledge

---

## Theory: Why This Works

### Compression Progress = Intelligence

**Schmidhuber**: Curiosity reward = improvement in compression ability
- Positive reward → Learned to compress better
- Maximizing → Maximizing learning curve steepness

### Utility Prevents Waste

**Problem**: Pure compression learns anything, useful or not

**Solution**: `Reward = Utility × Compression Progress`

**Result**:
- High utility + high compression → **Learn!**
- Low utility + high compression → **Ignore** (junk food)
- High utility + low compression → **Use tool**

### Topology Tunneling = Insight

**Problem**: No direct solution in knowledge base

**Solution**: Find structural analogy to known solution

**Example**: Car housing ≈ Jar lid (structurally)
→ Use prying technique from jar lid removal

---

## Performance

### Compression Scoring
- **Fast** (O(1)): Tool similarity, goal alignment
- **Medium** (O(n)): Semantic search, analogical matching
- **Slow** (O(n²)): Graph paths (depth-limited to 3)

### Memory per Node
- ThoughtNode: ~2KB (500B properties + 1.5KB embedding)
- Tool: ~3KB (pattern + embedding + metadata)

### Practical Limits
- ~50K nodes: Excellent
- ~500K nodes: Good
- ~5M nodes: Consider sharding

---

## References

- **Schmidhuber, J. (2009)** - "Driven by Compression Progress" arXiv:0812.4360
- **Meta AI (2024)** - COCONUT continuous thought models
- **Gen 3 Architecture** - This implementation

---

## Decision Tree

```
Problem arrives
├─ Has fast solution (tool)?
│  └─ Yes → Execute (System 1) ✓
└─ No → System 2:
   ├─ Deconstruct problem
   ├─ Calculate utility
   ├─ High utility?
   │  ├─ Yes → Explore:
   │  │  ├─ Find analogical tools
   │  │  ├─ Hypothesize (topology tunnel)
   │  │  ├─ Constrain (validate)
   │  │  └─ Compress to new tool
   │  └─ No → Ignore (low utility)
```