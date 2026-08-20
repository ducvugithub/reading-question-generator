"""
GNN encoder: converts KG subgraph into node embeddings via message passing.

Input:  NetworkX graph with node text features
Output: node embedding tensor per entity (shape: [num_nodes, hidden_dim])
Status: NOT YET IMPLEMENTED
"""


def encode_subgraph(graph, node_features: dict) -> dict:
    """Run GNN message passing over subgraph, return per-node embeddings."""
    raise NotImplementedError
