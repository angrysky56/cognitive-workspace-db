"""
Cognitive Workspace Database (CWD) MCP Server

A "System 2" reasoning engine that operates in latent space using Neo4j for structural
reasoning and Chroma for vector operations. Implements cognitive primitives based on
cutting-edge research (Meta COCONUT, Hierarchical Reasoning Models).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import ollama
import torch
from chromadb.config import Settings
from mcp.server import Server
from mcp.types import TextContent, Tool
from neo4j import GraphDatabase
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cwd-mcp")

# Initialize MCP server
server = Server("cognitive-workspace-db")


# ============================================================================
# Configuration
# ============================================================================

class CWDConfig(BaseSettings):
    """
    Configuration for Cognitive Workspace Database.

    Loads from environment variables or .env file - never hardcode credentials!
    Set NEO4J_PASSWORD in your .env file or environment.

    Searches for .env file in:
    1. Current directory
    2. Parent directory (project root when running from src/)
    """
    # Find .env file in project root (one level up from src/)
    _env_file = Path(__file__).parent.parent / ".env"

    model_config = SettingsConfigDict(
        env_file=str(_env_file) if _env_file.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(...)  # Required from environment - no default
    chroma_path: str = Field(default="./chroma_data")
    embedding_model: str = Field(default="qwen3-embedding:0.6b")
    confidence_threshold: float = Field(default=0.5)
    llm_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="qwen3:4b")


class CognitiveWorkspace:
    """
    Manages the cognitive workspace - a hybrid system combining:
    - Neo4j for structural/graph reasoning
    - Chroma for latent space vector operations
    - Cognitive primitives for active reasoning
    """

    def __init__(self, config: CWDConfig):
        self.config = config

        # Initialize Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )

        # Initialize Chroma
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=config.chroma_path,
            anonymized_telemetry=False
        ))

        # Initialize embedding model
        # For Qwen embeddings, optimize with flash_attention_2 if GPU available
        is_qwen = "qwen" in config.embedding_model.lower()
        has_gpu = torch.cuda.is_available()

        if is_qwen and has_gpu:
            try:
                # Try flash_attention_2 for GPU acceleration (requires flash-attn package)
                self.embedding_model = SentenceTransformer(
                    config.embedding_model,
                    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                    tokenizer_kwargs={"padding_side": "left"}
                )
                logger.info("Initialized Qwen embedding model with flash_attention_2 (GPU)")
            except Exception as e:
                # Fallback to standard (works on CPU and GPU without flash-attn)
                logger.info(f"flash_attention_2 not available, using standard attention: {e}")
                self.embedding_model = SentenceTransformer(
                    config.embedding_model,
                    tokenizer_kwargs={"padding_side": "left"}
                )
        elif is_qwen:
            # CPU mode - just use left padding for Qwen
            self.embedding_model = SentenceTransformer(
                config.embedding_model,
                tokenizer_kwargs={"padding_side": "left"}
            )
            logger.info("Initialized Qwen embedding model (CPU)")
        else:
            # Standard models (all-MiniLM, etc.)
            self.embedding_model = SentenceTransformer(config.embedding_model)

        # Create Chroma collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="thought_nodes",
            metadata={"description": "Cognitive workspace thought-nodes"}
        )

        logger.info("Cognitive Workspace initialized")

    def close(self):
        """Cleanup connections"""
        self.neo4j_driver.close()

    def _embed_text(self, text: str, is_query: bool = False) -> list[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed
            is_query: If True, uses query prompt for Qwen models (for similarity search).
                     If False, embeds as document (for storage).

        For Qwen embedding models, queries should use prompt_name="query" for better
        retrieval performance. Documents are embedded without prompts.
        """
        with torch.no_grad():
            # Check if this is a Qwen model that supports prompts
            is_qwen = "qwen" in self.config.embedding_model.lower()
            if is_qwen and is_query:
                # Use query prompt for better retrieval
                embedding = self.embedding_model.encode(text, prompt_name="query")
            else:
                # Standard document embedding
                embedding = self.embedding_model.encode(text)

            # Convert to list of floats
            if hasattr(embedding, 'tolist'):
                result = embedding.tolist()
            else:
                result = list(embedding)
            return result  # type: ignore[return-value]

    def _llm_generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 8000) -> str:
        """
        Generate text using local Ollama LLM.

        This is intentionally simple - the LLM only handles bridge text generation.
        Heavy lifting (reasoning, similarity, path-finding) happens in vector/graph operations.

        Framework context: The LLM is part of a cognitive workspace that combines:
        - Neo4j for structural reasoning (graph operations)
        - Chroma for latent space operations (vector similarity)
        - This LLM for generating human-readable bridge text

        The LLM's role is to produce concise, clear outputs - not to do the reasoning itself.
        """
        try:
            # Enhanced system prompt with framework context
            enhanced_system = f"""You are a text generation component in a cognitive reasoning system.

Your role: Generate concise, clear text outputs. The system handles reasoning via graph and vector operations.

Framework: Cognitive Workspace Database (System 2 Reasoning)
- Neo4j: Structural/graph reasoning
- Chroma: Vector similarity in latent space
- Your task: Bridge text generation only

{system_prompt}

CRITICAL: Output your final answer directly. You may think internally, but end with clear, concise output."""

            response = ollama.chat(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            )
            content = response["message"]["content"].strip()

            # Strip reasoning artifacts that models add
            # Remove <think>...</think> blocks
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<think>.*', '', content, flags=re.IGNORECASE)

            # Remove common reasoning prefixes (case-insensitive, at start of content)
            reasoning_patterns = [
                r'^(?:Okay|Alright|Let me|Hmm|So|Well|First|Now)\s*[,:]?\s*',
                r'^(?:The user|I need to|I should|Looking at)\s+.*?\.\s*',
            ]

            for pattern in reasoning_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE)

            # Extract actual content after reasoning markers
            # Look for patterns like "...reasoning... [OUTPUT] actual content" or similar
            output_markers = [
                r'(?:OUTPUT|ANSWER|RESULT|FINAL):\s*(.+)',  # Explicit markers
                r'\n\n(.+?)$',  # Last paragraph after double newline
            ]

            for marker_pattern in output_markers:
                match = re.search(marker_pattern, content, re.DOTALL | re.IGNORECASE)
                if match and len(match.group(1).strip()) > 20:
                    content = match.group(1).strip()
                    break

            # Clean up multiple spaces and newlines
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Max 2 newlines
            content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces
            content = content.strip()

            return content if content else "[No output generated]"
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return f"[LLM unavailable: {user_prompt[:50]}...]"

    def _create_thought_node(
        self,
        session,
        content: str,
        cognitive_type: str,
        parent_problem: str | None = None,
        confidence: float = 0.5,
        embedding: list[float] | None = None
    ) -> str:
        """
        Create a thought-node in both Neo4j and Chroma.

        If embedding is provided, uses it directly (e.g., for synthesis centroids).
        Otherwise generates embedding from content.
        """
        thought_id = f"thought_{int(time.time() * 1000000)}"
        if embedding is None:
            embedding = self._embed_text(content)

        # Store in Neo4j
        query = """
        CREATE (t:ThoughtNode {
            id: $id,
            content: $content,
            cognitive_type: $cognitive_type,
            confidence: $confidence,
            created_at: timestamp(),
            parent_problem: $parent_problem
        })
        RETURN t.id as id
        """
        result = session.run(
            query,
            id=thought_id,
            content=content,
            cognitive_type=cognitive_type,
            confidence=confidence,
            parent_problem=parent_problem
        )
        result.single()

        # Store embedding in Chroma
        self.collection.add(
            ids=[thought_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "cognitive_type": cognitive_type,
                "confidence": confidence,
                "parent_problem": parent_problem or ""
            }]
        )

        return thought_id

    # ========================================================================
    # Cognitive Primitive 1: Deconstruct
    # Breaks complex vectors into component thought-nodes with relationships
    # ========================================================================

    def deconstruct(self, problem: str, max_depth: int = 50) -> dict[str, Any]:
        """
        Break a complex problem into component thought-nodes.

        Creates a hierarchical graph representing problem decomposition:
        - Root node: Original problem
        - Child nodes: Sub-problems
        - Edges: DECOMPOSES_INTO relationships

        This operates similarly to Meta's COCONUT "continuous thought" but
        materializes the reasoning tree in a queryable graph structure.
        """
        logger.info(f"Deconstructing: {problem[:8000]}...")

        with self.neo4j_driver.session() as session:
            # Create root problem node
            root_id = self._create_thought_node(
                session,
                problem,
                "problem",
                parent_problem=None,
                confidence=1.0
            )

            # Simple decomposition (in production, use LLM)
            components = self._simple_decompose(problem)
            component_ids = []

            for comp in components:
                comp_id = self._create_thought_node(
                    session,
                    comp,
                    "sub_problem",
                    parent_problem=root_id,
                    confidence=0.8
                )
                component_ids.append(comp_id)

                # Create decomposition relationship
                session.run(
                    """
                    MATCH (parent:ThoughtNode {id: $parent_id})
                    MATCH (child:ThoughtNode {id: $child_id})
                    CREATE (parent)-[:DECOMPOSES_INTO]->(child)
                    """,
                    parent_id=root_id,
                    child_id=comp_id
                )

            # Get decomposition tree
            tree = self._get_decomposition_tree(session, root_id)

            return {
                "root_id": root_id,
                "component_ids": component_ids,
                "tree": tree,
                "message": f"Decomposed into {len(component_ids)} components"
            }

    def _simple_decompose(self, text: str) -> list[str]:
        """
        Break problem into 2-50 logical sub-components using LLM.

        The LLM handles decomposition logic while graph/vector ops handle the reasoning.
        """
        system_prompt = (
            "You decompose complex problems into 2-50 logical sub-components. "
            "Output ONLY the components as a numbered list (1., 2., 3., etc.). "
            "Each component should be a clear, actionable sub-problem. "
            "No explanations, no reasoning - just the list."
        )
        user_prompt = f"Decompose this problem into 2-50 logical sub-components:\n\n{text}\n\nComponents:"

        llm_output = self._llm_generate(system_prompt, user_prompt, max_tokens=8000)

        # Parse numbered list into components
        components = []
        for line in llm_output.split('\n'):
            line = line.strip()
            # Remove numbering like "1.", "1)", "•", etc.
            if line and len(line) > 3:
                # Strip common prefixes
                for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.',
                              '1)', '2)', '3)', '4)', '5)', '6)', '7)',
                              '•', '-', '*', '→', '►']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                if line and not line.startswith(('Ok', 'Here', 'The', 'I ')):
                    components.append(line)

        # Return at least one component (fallback to sentence split if parsing fails)
        if not components:
            components = [s.strip() for s in text.split('.') if s.strip()][:50]

        return components[:50]  # Cap at 50 components

    def _get_decomposition_tree(self, session, root_id: str) -> dict[str, Any]:
        """Retrieve decomposition tree"""
        result = session.run(
            """
            MATCH (root:ThoughtNode {id: $root_id})
            OPTIONAL MATCH (root)-[:DECOMPOSES_INTO]->(child:ThoughtNode)
            RETURN root, collect(child) as children
            """,
            root_id=root_id
        )
        record = result.single()
        if not record:
            return {}

        root = record["root"]
        children = record["children"]

        return {
            "id": root["id"],
            "content": root["content"],
            "type": root["cognitive_type"],
            "children": [{"id": c["id"], "content": c["content"]} for c in children]
        }

    # ========================================================================
    # Cognitive Primitive 2: Hypothesize
    # Discovers novel connections in latent space (similar to breadth-first
    # search in COCONUT but across graph + vector space)
    # ========================================================================

    def hypothesize(
        self,
        node_a_id: str,
        node_b_id: str,
        context: str | None = None
    ) -> dict[str, Any]:
        """
        Find novel connections between concepts in latent space.

        Combines:
        1. Graph path-finding (structural relationships)
        2. Vector similarity (semantic relationships)

        Creates a hypothesis node representing the discovered connection.
        """
        logger.info(f"Hypothesizing: {node_a_id} <-> {node_b_id}")

        with self.neo4j_driver.session() as session:
            # Get node contents
            nodes = session.run(
                """
                MATCH (a:ThoughtNode {id: $id_a})
                MATCH (b:ThoughtNode {id: $id_b})
                RETURN a.content as content_a, b.content as content_b
                """,
                id_a=node_a_id,
                id_b=node_b_id
            ).single()

            if not nodes:
                return {"error": "Nodes not found"}

            content_a = nodes["content_a"]
            content_b = nodes["content_b"]

            # Find graph paths
            paths = session.run(
                """
                MATCH path = (a:ThoughtNode {id: $id_a})-[*1..3]-(b:ThoughtNode {id: $id_b})
                RETURN path
                LIMIT 5
                """,
                id_a=node_a_id,
                id_b=node_b_id
            ).values()

            # Check vector similarity
            similarity_score = self._cosine_similarity(
                self._embed_text(content_a),
                self._embed_text(content_b)
            )

            # Generate hypothesis
            hypothesis_text = self._generate_hypothesis(
                content_a,
                content_b,
                similarity_score,
                context
            )

            # Create hypothesis node
            hyp_id = self._create_thought_node(
                session,
                hypothesis_text,
                "hypothesis",
                confidence=similarity_score
            )

            # Create relationships
            session.run(
                """
                MATCH (a:ThoughtNode {id: $id_a})
                MATCH (b:ThoughtNode {id: $id_b})
                MATCH (h:ThoughtNode {id: $hyp_id})
                CREATE (h)-[:CONNECTS {similarity: $similarity}]->(a)
                CREATE (h)-[:CONNECTS {similarity: $similarity}]->(b)
                """,
                id_a=node_a_id,
                id_b=node_b_id,
                hyp_id=hyp_id,
                similarity=similarity_score
            )

            return {
                "hypothesis_id": hyp_id,
                "hypothesis": hypothesis_text,
                "similarity": similarity_score,
                "path_count": len(paths),
                "message": "Hypothesis generated"
            }

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _generate_hypothesis(self, content_a: str, content_b: str,
                           similarity: float, context: str | None) -> str:
        """
        Generate hypothesis text explaining connection between concepts.

        LLM generates bridge text; similarity score comes from vector operations.
        """
        system_prompt = (
            "You explain connections between concepts in 1-2 clear sentences. "
            "Focus on the relationship, not the concepts themselves. "
            "Be insightful and concise. Output the explanation directly."
        )

        context_text = f"\nContext: {context}" if context else ""
        user_prompt = (
            f"Concept A: {content_a[:200]}\n"
            f"Concept B: {content_b[:200]}\n"
            f"Vector Similarity Score: {similarity:.2f} (computed by system){context_text}\n\n"
            f"Connection:"
        )

        return self._llm_generate(system_prompt, user_prompt, max_tokens=8000)

    # ========================================================================
    # Cognitive Primitive 3: Synthesize
    # Merges multiple vectors in latent space (similar to hierarchical
    # reasoning models' latent transformations)
    # ========================================================================

    def synthesize(
        self,
        node_ids: list[str],
        goal: str | None = None
    ) -> dict[str, Any]:
        """
        Merge multiple thought-nodes into a unified insight.

        Operates in latent space by:
        1. Computing centroid of input node embeddings
        2. Finding related concepts near the centroid
        3. Creating a synthesis node representing the merged insight

        This is analogous to "latent transformations" in HRMs.
        """
        logger.info(f"Synthesizing {len(node_ids)} nodes")

        with self.neo4j_driver.session() as session:
            # Get all node contents
            nodes_result = session.run(
                """
                MATCH (n:ThoughtNode)
                WHERE n.id IN $ids
                RETURN n.id as id, n.content as content
                """,
                ids=node_ids
            )

            nodes = {r["id"]: r["content"] for r in nodes_result}

            if len(nodes) < 2:
                return {"error": "Need at least 2 nodes to synthesize"}

            # Compute centroid in latent space (synthesis node lives at geometric center)
            embeddings = [self._embed_text(content) for content in nodes.values()]
            centroid = np.mean(embeddings, axis=0).tolist()

            # Generate synthesis (LLM merges only the input nodes - no external concepts)
            synthesis_text = self._generate_synthesis(list(nodes.values()), goal)

            # Create synthesis node with centroid embedding (latent space position)
            synth_id = self._create_thought_node(
                session,
                synthesis_text,
                "synthesis",
                confidence=0.7,
                embedding=centroid
            )

            # Create relationships
            for node_id in node_ids:
                session.run(
                    """
                    MATCH (source:ThoughtNode {id: $source_id})
                    MATCH (synth:ThoughtNode {id: $synth_id})
                    CREATE (synth)-[:SYNTHESIZES_FROM]->(source)
                    """,
                    source_id=node_id,
                    synth_id=synth_id
                )

            return {
                "synthesis_id": synth_id,
                "synthesis": synthesis_text,
                "source_count": len(node_ids),
                "message": "Synthesis created"
            }

    def _generate_synthesis(self, contents: list[str], goal: str | None) -> str:
        """
        Generate synthesis merging multiple concepts.

        LLM merges concept previews; centroid computation happens in vector space.
        """
        system_prompt = (
            "Given multiple concept fragments, write 1-2 sentences unifying them "
            "into a coherent insight. Be concise and focus on the common thread."
        )

        # Prepare concept previews (keep them short)
        previews = [f"- {c[:1500]}..." for c in contents[:50]]
        goal_text = f"\nGoal: {goal}" if goal else ""

        user_prompt = (
            f"Synthesize these {len(contents)} concepts:{goal_text}\n\n"
            + "\n".join(previews) +
            "\n\nUnified insight:"
        )

        return self._llm_generate(system_prompt, user_prompt, max_tokens=1500)

    # ========================================================================
    # Cognitive Primitive 4: Constrain
    # Validates thoughts against rules by projecting in latent space
    # ========================================================================

    def constrain(self, node_id: str, rules: list[str]) -> dict[str, Any]:
        """
        Apply constraints/rules to validate a thought-node.

        Projects the thought vector against rule vectors to check compatibility.
        This enables "checking work" as the human describes it - applying
        logical constraints to validate reasoning.
        """
        logger.info(f"Applying {len(rules)} constraints to {node_id}")

        with self.neo4j_driver.session() as session:
            # Get node content
            node = session.run(
                """
                MATCH (n:ThoughtNode {id: $id})
                RETURN n.content as content
                """,
                id=node_id
            ).single()

            if not node:
                return {"error": "Node not found"}

            content = node["content"]
            content_embedding = self._embed_text(content)  # Document embedding

            # Check each rule
            rule_results = []
            for rule in rules:
                # Rules act as queries against the document
                rule_embedding = self._embed_text(rule, is_query=True)
                similarity = self._cosine_similarity(content_embedding, rule_embedding)

                rule_results.append({
                    "rule": rule,
                    "score": similarity,
                    "satisfied": similarity > self.config.confidence_threshold
                })

            # Calculate overall score
            avg_score = np.mean([r["score"] for r in rule_results])
            all_satisfied = all(r["satisfied"] for r in rule_results)

            # Update node
            session.run(
                """
                MATCH (n:ThoughtNode {id: $id})
                SET n.constrained = true,
                    n.constraint_score = $score,
                    n.constraints_satisfied = $satisfied
                """,
                id=node_id,
                score=float(avg_score),
                satisfied=all_satisfied
            )

            return {
                "node_id": node_id,
                "overall_score": float(avg_score),
                "all_satisfied": all_satisfied,
                "rule_results": rule_results,
                "message": "Constraints applied"
            }


# ============================================================================
# MCP Tool Handlers
# ============================================================================

_workspace: CognitiveWorkspace | None = None


def get_workspace() -> CognitiveWorkspace:
    """Get or create workspace instance"""
    global _workspace
    if _workspace is None:
        # Config automatically loads from .env file or environment variables
        config = CWDConfig()  # type: ignore[call-arg]
        _workspace = CognitiveWorkspace(config)
    return _workspace


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available cognitive workspace tools"""
    return [
        Tool(
            name="deconstruct",
            description="Break a complex problem into component thought-nodes with hierarchical relationships. Creates a reasoning tree similar to Meta's COCONUT but materialized as a queryable graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": "The complex problem to decompose"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum decomposition depth",
                        "default": 3
                    }
                },
                "required": ["problem"]
            }
        ),
        Tool(
            name="hypothesize",
            description="Find novel connections between two concepts using both graph paths and vector similarity. Enables breadth-first search through conceptual space.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_a_id": {
                        "type": "string",
                        "description": "First thought-node ID"
                    },
                    "node_b_id": {
                        "type": "string",
                        "description": "Second thought-node ID"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context to guide hypothesis generation"
                    }
                },
                "required": ["node_a_id", "node_b_id"]
            }
        ),
        Tool(
            name="synthesize",
            description="Merge multiple thought-nodes into a unified insight by operating in latent space. Computes centroids and finds common patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of thought-node IDs to synthesize (minimum 2)"
                    },
                    "goal": {
                        "type": "string",
                        "description": "Optional goal to guide synthesis"
                    }
                },
                "required": ["node_ids"]
            }
        ),
        Tool(
            name="constrain",
            description="Apply constraints/rules to validate a thought-node by projecting against rule vectors. Enables 'checking work' through logical validation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Thought-node ID to constrain"
                    },
                    "rules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of constraint rules in natural language"
                    }
                },
                "required": ["node_id", "rules"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """Handle tool calls"""
    workspace = get_workspace()

    try:
        if name == "deconstruct":
            result = workspace.deconstruct(
                problem=arguments["problem"],
                max_depth=arguments.get("max_depth", 3)
            )
        elif name == "hypothesize":
            result = workspace.hypothesize(
                node_a_id=arguments["node_a_id"],
                node_b_id=arguments["node_b_id"],
                context=arguments.get("context")
            )
        elif name == "synthesize":
            result = workspace.synthesize(
                node_ids=arguments["node_ids"],
                goal=arguments.get("goal")
            )
        elif name == "constrain":
            result = workspace.constrain(
                node_id=arguments["node_id"],
                rules=arguments["rules"]
            )
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
