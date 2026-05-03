"""
SW-KG: Small-World Knowledge Graph
===================================
Standalone implementation for multi-agent LLM coordination.

Usage:
    from swkg import KnowledgeGraph
    
    kg = KnowledgeGraph()
    kg.write("insight content", "insight", agent_id=0, token_cost=25, epoch=0)
    relevant = kg.read("insight", limit=3)
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional

# Composite SW score weights
SW_ALPHA = 0.40   # betweenness centrality (cross-cluster connectors)
SW_BETA = 0.30    # degree centrality (hub-ness)
SW_GAMMA = 0.20   # token_value (existing economy)
SW_DELTA = 0.10   # clustering coefficient (penalize local-only nodes)

# Default small-world parameters
DEFAULT_SW_K = 4      # Watts-Strogatz neighbors
DEFAULT_SW_P = 0.3    # Rewiring probability
DEFAULT_HUB_THRESHOLD = 3  # Reads before hub promotion


class KnowledgeGraph:
    """
    Small-World Knowledge Graph for multi-agent coordination.
    
    Combines three mechanisms:
    1. Watts-Strogatz small-world topology (O(log N) retrieval)
    2. Hub compression (frequent nodes bypass filtering)
    3. Token economy (value = cost × reads)
    
    Example:
        >>> kg = KnowledgeGraph()
        >>> kg.write("Trade hubs concentrate in China, US, Germany",
        ...          node_type="insight", agent_id=0, token_cost=25, epoch=0)
        >>> relevant = kg.read(query_type="insight", limit=3)
        >>> print(len(relevant))  # Returns up to 3 most relevant nodes
    """
    
    def __init__(
        self,
        initial_nodes: int = 30,
        k: int = DEFAULT_SW_K,
        p: float = DEFAULT_SW_P,
        hub_threshold: int = DEFAULT_HUB_THRESHOLD
    ):
        """
        Initialize knowledge graph with small-world topology.
        
        Args:
            initial_nodes: Initial graph size (Watts-Strogatz n parameter)
            k: Number of nearest neighbors in ring topology
            p: Probability of rewiring each edge
            hub_threshold: Read count threshold for hub promotion
        """
        self.G = nx.watts_strogatz_graph(initial_nodes, k, p)
        self.nodes: Dict[str, Dict] = {}
        self.hubs = set()
        self.n_id = 0
        self.epoch = 0
        self._sw_scores: Dict[str, float] = {}
        self.k = k
        self.hub_threshold = hub_threshold
    
    def write(
        self,
        content: str,
        node_type: str,
        agent_id: int,
        token_cost: int,
        epoch: int
    ) -> str:
        """
        Write a new knowledge node to the graph.
        
        Args:
            content: Text content of the knowledge
            node_type: Type/category (e.g., "insight", "hypothesis", "evidence")
            agent_id: ID of agent that produced this knowledge
            token_cost: Token count of this content
            epoch: Current epoch number
            
        Returns:
            Node ID (string)
        """
        nid = f"n{self.n_id}"
        self.n_id += 1
        
        self.nodes[nid] = {
            "content": content,
            "type": node_type,
            "token_cost": token_cost,
            "token_value": 0,
            "epoch": epoch,
            "produced_by": agent_id,
            "reads": 0,
        }
        
        # Add to graph and connect to recent neighbors (small-world growth)
        self.G.add_node(nid)
        recent = list(self.nodes.keys())[-min(self.k, len(self.nodes)):]
        for r in recent:
            if r != nid:
                self.G.add_edge(nid, r)
        
        return nid
    
    def read(self, query_type: str, limit: int = 3) -> List[Tuple[str, str]]:
        """
        Retrieve most relevant knowledge nodes.
        
        Uses composite SW-Score ranking:
            score = α·betweenness + β·degree + γ·token_value - δ·clustering
        
        Hubs are always included if capacity allows.
        
        Args:
            query_type: Type of knowledge to retrieve
            limit: Maximum number of nodes to return
            
        Returns:
            List of (node_id, content) tuples
        """
        candidates = []
        for nid, node in self.nodes.items():
            if node["type"] == query_type or nid in self.hubs:
                candidates.append((nid, node))
        
        # Sort by SW-Score (if available) or token_value (cold-start fallback)
        if self._sw_scores:
            candidates.sort(
                key=lambda x: (
                    x[0] in self.hubs,  # Hubs always rank first
                    self._sw_scores.get(x[0], 0.0)
                ),
                reverse=True
            )
        else:
            candidates.sort(
                key=lambda x: (x[0] in self.hubs, x[1]["token_value"]),
                reverse=True
            )
        
        selected = candidates[:limit]
        
        # Update token economy on read
        for nid, node in selected:
            node["reads"] += 1
            node["token_value"] += node["token_cost"]
            if node["reads"] >= self.hub_threshold and nid not in self.hubs:
                self.hubs.add(nid)
        
        return [(nid, node["content"]) for nid, node in selected]
    
    def promote_hubs(self):
        """
        Promote frequently-accessed nodes to hub status.
        
        Called at epoch boundaries. Recomputes SW-Scores for all nodes.
        """
        for nid, node in self.nodes.items():
            if node["reads"] >= self.hub_threshold:
                self.hubs.add(nid)
        
        self._recompute_sw_scores()
    
    def _recompute_sw_scores(self):
        """
        Compute composite Small-World score for every knowledge node.
        
        Score = α·betweenness + β·degree + γ·token_value_norm - δ·clustering
        
        All components normalized to [0, 1] across current node set.
        Uses NetworkX graph metrics (cached — only called at epoch boundaries).
        """
        knowledge_nids = list(self.nodes.keys())
        if len(knowledge_nids) < 2:
            self._sw_scores = {nid: 0.0 for nid in knowledge_nids}
            return
        
        # Build subgraph containing only knowledge nodes
        subgraph = self.G.subgraph([n for n in self.G.nodes if n in self.nodes])
        
        # NetworkX centrality (approximate betweenness for speed)
        k_sample = min(10, len(subgraph))
        try:
            betweenness = nx.betweenness_centrality(
                subgraph, normalized=True, k=k_sample
            )
        except Exception:
            betweenness = {n: 0.0 for n in subgraph.nodes}
        
        degree = nx.degree_centrality(subgraph)
        
        try:
            clustering = nx.clustering(subgraph)
        except Exception:
            clustering = {n: 0.0 for n in subgraph.nodes}
        
        # Normalize token_value
        tv_vals = [self.nodes[nid]["token_value"] for nid in knowledge_nids]
        tv_max = max(tv_vals) if max(tv_vals) > 0 else 1.0
        
        # Compute composite scores
        for nid in knowledge_nids:
            tv_norm = self.nodes[nid]["token_value"] / tv_max
            score = (
                SW_ALPHA * betweenness.get(nid, 0.0)
                + SW_BETA * degree.get(nid, 0.0)
                + SW_GAMMA * tv_norm
                - SW_DELTA * clustering.get(nid, 0.0)
            )
            self._sw_scores[nid] = score
    
    def context_tokens_for(
        self,
        query_type: str,
        cached_nodes: Optional[List[Tuple[str, str]]] = None
    ) -> int:
        """
        Estimate tokens needed to load relevant context.
        
        Args:
            query_type: Type of knowledge to retrieve
            cached_nodes: Pre-fetched nodes (avoids second read call)
            
        Returns:
            Estimated token count
        """
        nodes = cached_nodes if cached_nodes is not None else self.read(query_type)
        # Rough estimate: 1.3 tokens per word
        total = sum(len(c.split()) * 1.3 for _, c in nodes)
        return max(int(total), 75)
    
    def get_stats(self) -> Dict:
        """
        Get current graph statistics.
        
        Returns:
            Dictionary with node count, hub count, average degree, etc.
        """
        return {
            "total_nodes": len(self.nodes),
            "hubs": len(self.hubs),
            "avg_degree": sum(dict(self.G.degree()).values()) / len(self.G.nodes)
            if len(self.G.nodes) > 0
            else 0,
            "total_reads": sum(n["reads"] for n in self.nodes.values()),
        }


# Example usage
if __name__ == "__main__":
    kg = KnowledgeGraph()
    
    # Simulate multi-agent knowledge accumulation
    insights = [
        "Trade networks exhibit hub concentration in China, US, Germany",
        "Supply chain resilience correlates with network redundancy",
        "Regional clusters show geopolitical alignment patterns",
        "Cascade failures propagate through high-betweenness nodes",
    ]
    
    for i, insight in enumerate(insights):
        nid = kg.write(
            content=insight,
            node_type="insight",
            agent_id=i % 3,  # 3 agents
            token_cost=len(insight.split()) * 2,  # rough token estimate
            epoch=0
        )
        print(f"Wrote {nid}: {insight[:50]}...")
    
    # Retrieve relevant knowledge
    print("\nRetrieving top 3 insights:")
    relevant = kg.read(query_type="insight", limit=3)
    for nid, content in relevant:
        print(f"  {nid}: {content}")
    
    # Promote hubs and check stats
    kg.promote_hubs()
    stats = kg.get_stats()
    print(f"\nGraph stats: {stats}")
