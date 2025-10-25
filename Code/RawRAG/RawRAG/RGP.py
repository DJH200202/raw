import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class RGP(nn.Module):
    """
    Random walk Graph Propagation for entity-aware retrieval.
    
    This module performs random walks on a heterogeneous graph containing both 
    text chunks and entities to identify relevant documents for answering questions.
    """
    
    def __init__(self, device, iter=3, factor=1, prob=0.85):
        """
        Initialize RGP module.
        
        Args:
            device: Device for computation ("cuda" or "cpu")
            iter: Number of random walk iterations
            factor: Scaling factor for sparse attention (controls sparsity level)
            prob: Random walk restart probability (probability of staying vs. jumping)
        """
        super(RGP, self).__init__()
        self.device = device  
        self.factor = factor  # Controls sparsity in attention maps
        self.iter = iter  # Number of propagation steps
        self.prob = prob  # Restart probability for random walk


    def Generate_StartPoint(self, device, Batch_size, chunk_size, entity_size):
        """
        Generate starting distribution for random walk.
        
        Args:
            device: Computation device
            Batch_size: Number of examples in batch
            chunk_size: Number of text chunks in graph
            entity_size: Number of entities in graph
            
        Returns:
            Initial probability distribution with all mass at the start node
        """
        # Create probability vector: [start_node, chunks..., entities...]
        e = torch.zeros([Batch_size, chunk_size+entity_size+1], dtype=torch.float32, device=device)
        e[:, 0] = 1  # All probability mass starts at the virtual start node
        e = e.unsqueeze(-1)  # Add dimension for matrix multiplication
        return e

    def get_attention_map(self, chunk_to_chunk, chunk_to_entity, chunk_question_probs, entity_question_probs, device, top_k=100, use_sparse=False):
        """
        Construct heterogeneous attention/transition matrix for random walk.
        
        This creates a transition matrix for a graph containing:
        - A virtual start node (index 0)
        - Text chunks (indices 1 to chunk_size)
        - Entities (indices chunk_size+1 to chunk_size+entity_size)
        
        Args:
            chunk_to_chunk: Similarity matrix between text chunks
            chunk_to_entity: Association matrix between chunks and entities
            chunk_question_probs: Question-chunk relevance scores
            entity_question_probs: Question-entity relevance scores
            device: Computation device
            top_k: Number of top connections to keep (for sparsity)
            use_sparse: Whether to apply sparsity constraints
            
        Returns:
            Transition matrix for random walk with softmax-normalized rows
        """
        batch_size, chunk_size, _ = chunk_to_chunk.shape
        _, _, entity_size = chunk_to_entity.shape

        # Initialize transition matrix with -inf (will become 0 after softmax)
        SpatialMap = torch.ones([batch_size, chunk_size+entity_size+1, chunk_size+entity_size+1], device=device) * -torch.inf
        
        # Convert numpy arrays to tensors if needed
        if isinstance(chunk_question_probs, np.ndarray):
            chunk_question_probs = torch.tensor(chunk_question_probs, device=device)
        
        if isinstance(entity_question_probs, np.ndarray):
            entity_question_probs = torch.tensor(entity_question_probs, device=device)
        
        # Process chunk-entity associations
        chunk_to_entity = torch.exp(chunk_to_entity)
        # Set disconnected pairs to -inf for softmax
        chunk_to_entity[chunk_to_entity == 0] = -torch.inf

        # Apply sparsity constraints if enabled
        if use_sparse:
            # Ensure top_k doesn't exceed matrix dimensions
            top_k = min(top_k, chunk_to_chunk.size(-2)) 
            # Keep only top-k connections in chunk-chunk similarity
            index = chunk_to_chunk < chunk_to_chunk.topk(top_k, dim=-2)[0][..., -1:, :]
            inf_idx = torch.isinf(chunk_to_chunk)
            chunk_to_chunk[index] = 0
            # Preserve -inf values to avoid NaN
            chunk_to_chunk[inf_idx] = 0

            # Ensure symmetry in chunk-chunk connections
            chunk_to_chunk = torch.add(chunk_to_chunk, chunk_to_chunk.transpose(-1, -2)) / 2
            
            # Convert zeros back to -inf for proper softmax behavior
            chunk_to_chunk[chunk_to_chunk == 0] = -torch.inf

            # Apply sparsity to question-chunk probabilities
            for b in range(batch_size):
                non_inf_mask = chunk_question_probs[b, :] < chunk_question_probs[b, :].topk(top_k, dim=-1)[0][..., -1:]
                chunk_question_probs[b, non_inf_mask] = -torch.inf
        
        # Fill transition matrix with connections:
        # Start node to chunks and entities (based on question relevance)
        SpatialMap[:, 1:chunk_size+1, 0] = chunk_question_probs 
        SpatialMap[:, 0, 1:chunk_size+1] = chunk_question_probs

        SpatialMap[:, chunk_size+1:, 0] = entity_question_probs
        SpatialMap[:, 0, chunk_size+1:] = entity_question_probs

        # Chunk-to-chunk connections (semantic similarity)
        SpatialMap[:, 1:chunk_size+1, 1:chunk_size+1] = chunk_to_chunk

        # Chunk-entity bidirectional connections
        SpatialMap[:, 1:chunk_size+1, chunk_size+1:] = chunk_to_entity
        SpatialMap[:, chunk_size+1:, 1:chunk_size+1] = chunk_to_entity.transpose(-1, -2)
        
        # Apply softmax to get proper transition probabilities
        SpatialMap = torch.softmax(SpatialMap, dim=-2) 
        SpatialMap = torch.nan_to_num(SpatialMap, nan=0.0)  # Handle any NaN values
        return SpatialMap

    def random_walk(self, prob, e, transition, pre_d):
        """
        Perform one step of random walk with restart.
        
        Args:
            prob: Probability of continuing walk (vs. restarting)
            e: Restart distribution (where to restart from)
            transition: Transition matrix for the graph
            pre_d: Previous step distribution
            
        Returns:
            Updated probability distribution after one random walk step
        """
        # Random walk equation: d_t = prob * A * d_{t-1} + (1-prob) * e
        d = prob * torch.matmul(transition, pre_d) + (1 - prob) * e  
        return d

    def random_walk_analytic(self, prob, e, transition):
        """
        Compute steady-state distribution analytically (closed-form solution).
        
        Args:
            prob: Probability of continuing walk
            e: Restart distribution
            transition: Transition matrix
            
        Returns:
            Steady-state distribution from analytic solution
        """
        # Solve: d = prob * A * d + (1-prob) * e
        # Rearranging: (I - prob * A) * d = (1-prob) * e
        # Solution: d = (1-prob) * (I - prob * A)^{-1} * e
        rev = torch.linalg.inv(torch.eye(transition.size(-2)).to(transition.device) - prob * transition)
        d = (1 - prob) * rev * e
        return d

    def forward(self, chunk_to_chunk, chunk_to_entity, chunk_question_probs, entity_question_probs, use_analytic=False, use_sparse=False):
        """
        Forward pass: perform random walk on heterogeneous graph for retrieval.
        
        Args:
            chunk_to_chunk: Text chunk similarity matrix [batch, chunks, chunks]
            chunk_to_entity: Chunk-entity association matrix [batch, chunks, entities]
            chunk_question_probs: Question-chunk relevance scores [batch, chunks]
            entity_question_probs: Question-entity relevance scores [batch, entities]
            use_analytic: Whether to use analytic solution (faster but may be unstable)
            use_sparse: Whether to apply sparsity constraints
            
        Returns:
            Tuple of (final_distribution, transition_matrix, walk_path)
            - final_distribution: Final probability distribution over nodes
            - transition_matrix: The constructed transition matrix
            - walk_path: List of distributions at each walk step (empty if analytic)
        """
        batch_size, chunk_size, _ = chunk_to_chunk.shape
        _, _, entity_size = chunk_to_entity.shape

        # Generate initial distribution (all mass at start node)
        d = self.Generate_StartPoint(device=chunk_to_chunk.device, Batch_size=batch_size, chunk_size=chunk_size, entity_size=entity_size)
        e = d  # Restart distribution (same as initial)
        
        # Construct heterogeneous transition matrix
        A = self.get_attention_map(chunk_to_chunk, chunk_to_entity, chunk_question_probs, entity_question_probs, device=chunk_to_chunk.device, top_k=self.factor * 5, use_sparse=use_sparse)
        
        walk_path = []  # Track walk progression
        
        if use_analytic:
            # Use closed-form solution for steady state
            d = self.random_walk_analytic(prob=self.prob, e=e, transition=A)
        else:
            # Iterative random walk steps
            walk_path.append(d.clone())  # Store initial distribution
            for i in range(self.iter):
                d = self.random_walk(prob=self.prob, e=e, transition=A, pre_d=d) 
                walk_path.append(d.clone())  # Store intermediate distributions

        # Return final distribution (transposed for convenience), transition matrix, and walk path
        return d.transpose(2, 1).contiguous(), A, walk_path
