import torch
import numpy as np

def masking(node_features, adj, mask_ratio=0.1, mask_edge=0.1):
    """
    Apply masking to node features and adjacency matrix.
    
    Args:
        node_features: torch.Tensor, original node feature matrix [num_nodes, feature_dim]
        adj: torch.Tensor, adjacency matrix [num_nodes, num_nodes]
        mask_ratio: float, ratio of node features to mask (0.0 to 1.0)
        mask_edge: float, ratio of edges to mask (0.0 to 1.0)
    
    Returns:
        masked_features: torch.Tensor, masked node feature matrix
        masked_adj: torch.Tensor, masked adjacency matrix
    """
    # Convert to tensors if numpy arrays
    if isinstance(node_features, np.ndarray):
        node_features = torch.from_numpy(node_features)
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    
    # Make copies to avoid modifying original data
    masked_features = node_features.clone()
    masked_adj = adj.clone()
    
    num_nodes = node_features.shape[0]
    
    # Initialize mask indices
    mask_indices = torch.tensor([], dtype=torch.long, device=node_features.device)
    mask_edge_indices = torch.tensor([], dtype=torch.long, device=adj.device)
    
    # Mask node features
    if mask_ratio > 0:
        num_mask_nodes = int(num_nodes * mask_ratio)
        mask_indices = torch.randperm(num_nodes)[:num_mask_nodes]
        masked_features[mask_indices] = 0.0
    
    # Mask edges
    if mask_edge > 0:
        # Find existing edges (non-zero entries)
        edge_indices = torch.nonzero(adj, as_tuple=False)
        num_edges = edge_indices.shape[0]
        
        if num_edges > 0:
            num_mask_edges = int(num_edges * mask_edge)
            # Randomly select edges to mask
            mask_edge_indices = torch.randperm(num_edges)[:num_mask_edges]
            
            # Mask selected edges
            for idx in mask_edge_indices:
                i, j = edge_indices[idx]
                masked_adj[i, j] = 0.0
                if adj.shape[0] == adj.shape[1]:  # If square matrix, mask symmetric entry
                    masked_adj[j, i] = 0.0
    
    return masked_features, masked_adj, mask_indices, mask_edge_indices

def compute_loss(original_features, original_adj, reconstructed_features, reconstructed_adj, 
                mask_indices, mask_edge_indices, lambda_attr=1.0, lambda_struct=1.0):
    """
    Compute total reconstruction loss for RAGFormer model.
    
    Args:
        original_features: torch.Tensor, original node features [num_nodes, feature_dim]
        original_adj: torch.Tensor, original adjacency matrix [num_nodes, num_nodes]
        reconstructed_features: torch.Tensor, reconstructed node features [num_nodes, feature_dim]
        reconstructed_adj: torch.Tensor, reconstructed adjacency matrix [num_nodes, num_nodes]
        mask_indices: torch.Tensor, indices of masked nodes
        mask_edge_indices: torch.Tensor, indices of masked edges
        lambda_attr: float, weight for attribute reconstruction loss
        lambda_struct: float, weight for structure reconstruction loss
    
    Returns:
        total_loss: torch.Tensor, total weighted loss
        attr_loss: torch.Tensor, attribute reconstruction loss
        struct_loss: torch.Tensor, structure reconstruction loss
    """
    # Attribute reconstruction loss (MSE on masked nodes only)
    if len(mask_indices) > 0:
        attr_loss = torch.nn.functional.mse_loss(
            reconstructed_features[mask_indices], 
            original_features[mask_indices]
        )
    else:
        attr_loss = torch.tensor(0.0, device=original_features.device)
    
    # Structure reconstruction loss (BCE on masked edges only)
    if len(mask_edge_indices) > 0:
        # Get original and reconstructed values for masked edges
        original_masked_edges = []
        reconstructed_masked_edges = []
        
        for idx in mask_edge_indices:
            i, j = idx
            original_masked_edges.append(original_adj[i, j])
            reconstructed_masked_edges.append(reconstructed_adj[i, j])
        
        original_masked_edges = torch.stack(original_masked_edges)
        reconstructed_masked_edges = torch.stack(reconstructed_masked_edges)
        
        # Binary cross-entropy loss
        struct_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            reconstructed_masked_edges, 
            original_masked_edges
        )
    else:
        struct_loss = torch.tensor(0.0, device=original_adj.device)
    
    # Total weighted loss
    total_loss = lambda_attr * attr_loss + lambda_struct * struct_loss
    
    return total_loss, attr_loss, struct_loss