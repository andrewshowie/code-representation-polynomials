"""
Hybrid Structural-Semantic Code Representation

This module integrates dependency tree polynomials with neural code embeddings to create
hybrid representations that capture both structural and semantic properties of code.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from scipy.spatial.distance import cosine
import ot
import matplotlib.pyplot as plt
import os

# Part 1: Neural Code Embeddings
# ----------------------------

class CodeEmbedder:
    """Generate semantic embeddings for code using pre-trained models."""
    
    def __init__(self, model_name="microsoft/codebert-base", use_cache=True):
        """
        Initialize with a pre-trained code model.
        
        Args:
            model_name: Name of a pre-trained model from HuggingFace
            use_cache: Whether to cache embeddings for reuse
        """
        self.model_name = model_name
        self.embedding_dim = 768  # Typical dimension for CodeBERT
        self.use_cache = use_cache
        self.embedding_cache = {}
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Model loading error: {e}")
            print("Using simulated embeddings for demonstration")
            self.tokenizer = None
            self.model = None
    
    def embed_code(self, code: str) -> np.ndarray:
        """
        Generate an embedding vector for the given code snippet.
        
        Args:
            code: Source code string
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        # Check cache first
        if self.use_cache and code in self.embedding_cache:
            return self.embedding_cache[code]
            
        if self.model is not None and self.tokenizer is not None:
            try:
                # Use the actual model if available
                inputs = self.tokenizer(code, 
                                       return_tensors="pt", 
                                       truncation=True, 
                                       max_length=512,
                                       padding="max_length")
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Multiple options for code representation:
                # 1. Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                
                # 2. Average all token embeddings (often works better for code)
                # Create attention mask to ignore padding tokens
                attention_mask = inputs['attention_mask'].numpy().flatten()
                token_embeddings = outputs.last_hidden_state[0].numpy()
                # Apply mask and compute mean over non-padding tokens
                masked_embeddings = token_embeddings * attention_mask[:, np.newaxis]
                sum_embeddings = np.sum(masked_embeddings, axis=0)
                sum_mask = np.sum(attention_mask) + 1e-10  # Avoid division by zero
                mean_embedding = sum_embeddings / sum_mask
                
                # Combine approaches (can be tuned based on performance)
                embedding = 0.7 * cls_embedding + 0.3 * mean_embedding
                
                # Store in cache
                if self.use_cache:
                    self.embedding_cache[code] = embedding
                    
                return embedding
                
            except Exception as e:
                print(f"Embedding error: {e}")
                print("Falling back to simulated embeddings")
                # Fall back to simulated embedding
        
        # Simulate an embedding for demonstration or fallback
        # This creates a deterministic but unique embedding based on code content
        seed = sum(ord(c) for c in code)
        np.random.seed(seed)
        embedding = np.random.randn(self.embedding_dim)
        
        # Normalize the embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        # Store in cache
        if self.use_cache:
            self.embedding_cache[code] = embedding
            
        return embedding
    
    def embed_function(self, code: str, function_name: str = None) -> np.ndarray:
        """
        Extract and embed a specific function from the code.
        
        Args:
            code: Source code containing the function
            function_name: Name of the function to extract (or None for the whole file)
            
        Returns:
            numpy.ndarray: Embedding vector for the function
        """
        if function_name is None:
            return self.embed_code(code)
        
        try:
            # Try to use proper parsing with ast
            import ast
            
            try:
                parsed = ast.parse(code)
                for node in ast.walk(parsed):
                    if isinstance(node, ast.FunctionDef) and node.name == function_name:
                        # Get source lines for the function
                        func_lines = code.splitlines()[node.lineno-1:node.end_lineno]
                        func_code = '\n'.join(func_lines)
                        return self.embed_code(func_code)
            except (SyntaxError, TypeError):
                # If AST parsing fails, fall back to regex
                pass
                
            # Fall back to regex-based extraction
            import re
            pattern = r'def\s+' + function_name + r'\s*\(.*?\).*?(?=def|\Z)'
            match = re.search(pattern, code, re.DOTALL)
            
            if match:
                function_code = match.group(0)
                return self.embed_code(function_code)
            else:
                raise ValueError(f"Function {function_name} not found in code")
                
        except Exception as e:
            print(f"Function extraction error: {e}")
            # Return embedding of the whole file as fallback
            return self.embed_code(code)
    
    def compute_semantic_similarity(self, code1: str, code2: str) -> float:
        """
        Compute the semantic similarity between two code snippets.
        
        Args:
            code1: First code snippet
            code2: Second code snippet
            
        Returns:
            float: Similarity score (0-1)
        """
        embedding1 = self.embed_code(code1)
        embedding2 = self.embed_code(code2)
        
        # Normalize embeddings
        norm1 = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        norm2 = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        # Compute cosine similarity
        similarity = 1.0 - cosine(norm1, norm2)
        
        return similarity
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache = {}
        print("Embedding cache cleared")

# Part 2: Hybrid Representation
# --------------------------

class HybridCodeRepresentation:
    """Combine structural (polynomial) and semantic (embedding) representations."""
    
    def __init__(self, embedder=None):
        self.embedder = embedder or CodeEmbedder()
        
    def create_hybrid_representation(self, 
                                    code: str, 
                                    polynomial_coefficients: np.ndarray, 
                                    alpha: float = 0.5) -> Dict[str, Any]:
        """
        Create a hybrid representation combining structure and semantics.
        
        Args:
            code: Source code string
            polynomial_coefficients: Coefficient vector from dependency tree polynomial
            alpha: Weight for balancing structural vs semantic components (0-1)
            
        Returns:
            dict: Hybrid representation containing both components and combined metrics
        """
        # Generate semantic embedding
        semantic_embedding = self.embedder.embed_code(code)
        
        # Normalize both representations
        poly_norm = polynomial_coefficients / (np.linalg.norm(polynomial_coefficients) + 1e-10)
        sem_norm = semantic_embedding / (np.linalg.norm(semantic_embedding) + 1e-10)
        
        # Store both components
        hybrid = {
            'structural': polynomial_coefficients,
            'semantic': semantic_embedding,
            'structural_normalized': poly_norm,
            'semantic_normalized': sem_norm,
            'alpha': alpha,
            'code': code  # Store code for reference
        }
        
        return hybrid
    
    def compute_hybrid_distance(self, 
                               hybrid1: Dict[str, Any], 
                               hybrid2: Dict[str, Any], 
                               alpha: float = None) -> Dict[str, float]:
        """
        Compute distance between two hybrid representations.
        
        Args:
            hybrid1: First hybrid representation
            hybrid2: Second hybrid representation
            alpha: Optional override for the alpha weight
            
        Returns:
            dict: Distance metrics (structural, semantic, and combined)
        """
        # Use provided alpha or default to the one in the hybrid
        if alpha is None:
            alpha = hybrid1.get('alpha', 0.5)
        
        # Compute structural distance using optimal transport
        struct_dist = self._compute_ot_distance(
            hybrid1['structural_normalized'],
            hybrid2['structural_normalized']
        )
        
        # Compute semantic distance using cosine distance
        semantic_dist = cosine(hybrid1['semantic_normalized'], hybrid2['semantic_normalized'])
        
        # Combine distances
        combined_dist = alpha * struct_dist + (1 - alpha) * semantic_dist
        
        return {
            'structural_distance': struct_dist,
            'semantic_distance': semantic_dist,
            'combined_distance': combined_dist,
            'alpha': alpha
        }
    
    def _compute_ot_distance(self, coef1: np.ndarray, coef2: np.ndarray) -> float:
        """
        Compute optimal transport distance between coefficient vectors.
        
        Args:
            coef1: First coefficient vector
            coef2: Second coefficient vector
            
        Returns:
            float: OT distance
        """
        # Ensure equal length
        max_len = max(len(coef1), len(coef2))
        c1 = np.zeros(max_len)
        c2 = np.zeros(max_len)
        c1[:len(coef1)] = coef1
        c2[:len(coef2)] = coef2
        
        # Normalize to create distributions
        p = c1 / (np.sum(c1) + 1e-10)
        q = c2 / (np.sum(c2) + 1e-10)
        
        # Compute cost matrix based on index distance
        M = np.abs(np.subtract.outer(np.arange(max_len), np.arange(max_len)))
        M /= M.max() + 1e-10  # Normalize
        
        # Compute Wasserstein distance using Sinkhorn algorithm
        epsilon = 0.1  # Regularization parameter
        ot_distance = ot.sinkhorn2(p, q, M, epsilon)
        
        return float(ot_distance)
    
    def visualize_distance_space(self, 
                                hybrids: List[Dict[str, Any]], 
                                labels: List[str],
                                title: str = "Code Representations in Hybrid Distance Space"):
        """
        Visualize code snippets in 2D distance space.
        
        Args:
            hybrids: List of hybrid representations
            labels: List of labels for each representation
            title: Title for the visualization
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        n = len(hybrids)
        distances = np.zeros((n, n, 2))
        
        # Compute all pairwise distances
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = self.compute_hybrid_distance(hybrids[i], hybrids[j])
                    distances[i, j, 0] = dist['structural_distance']
                    distances[i, j, 1] = dist['semantic_distance']
        
        # Average distances from each point to all others
        avg_distances = np.mean(distances, axis=1)
        
        # Create 2D visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            avg_distances[:, 0], 
            avg_distances[:, 1],
            c=np.arange(n),  # Color points by index
            cmap='viridis',
            s=100,  # Point size
            alpha=0.8
        )
        
        # Add labels with a slight offset for readability
        for i, label in enumerate(labels):
            ax.annotate(
                label, 
                (avg_distances[i, 0] + 0.01, avg_distances[i, 1] + 0.01),
                fontsize=10,
                alpha=0.8
            )
        
        # Add connecting lines to show distances
        for i in range(n):
            for j in range(i+1, n):
                ax.plot(
                    [avg_distances[i, 0], avg_distances[j, 0]],
                    [avg_distances[i, 1], avg_distances[j, 1]],
                    'k-',
                    alpha=0.2
                )
        
        ax.set_xlabel("Structural Distance", fontsize=12)
        ax.set_ylabel("Semantic Distance", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add legend and colorbar
        plt.colorbar(scatter, ax=ax, label="Sample Index")
        
        return fig
    
    def visualize_hybrid_similarity_matrix(self, 
                                          hybrids: List[Dict[str, Any]], 
                                          labels: List[str],
                                          alpha: float = None):
        """
        Visualize similarity matrix between multiple hybrid representations.
        
        Args:
            hybrids: List of hybrid representations
            labels: List of labels for each representation
            alpha: Weight for structural vs semantic components
            
        Returns:
            matplotlib.figure.Figure: Similarity matrix visualization
        """
        n = len(hybrids)
        sim_matrix = np.zeros((n, n))
        
        # Compute similarity matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    dist = self.compute_hybrid_distance(hybrids[i], hybrids[j], alpha)
                    # Convert distance to similarity (1 / (1 + distance))
                    sim_matrix[i, j] = 1.0 / (1.0 + dist['combined_distance'])
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
        
        # Add labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = fig.colorbar(im)
        cbar.set_label('Similarity (1 / (1 + distance))')
        
        # Add title
        current_alpha = alpha if alpha is not None else hybrids[0].get('alpha', 0.5)
        ax.set_title(f'Hybrid Similarity Matrix (α={current_alpha:.2f})')
        
        # Add similarity values in cells
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                              ha="center", va="center", color="w" if sim_matrix[i, j] < 0.7 else "black")
        
        fig.tight_layout()
        return fig
    
    def visualize_ot_plan(self, 
                         hybrid1: Dict[str, Any], 
                         hybrid2: Dict[str, Any],
                         title: str = None):
        """
        Visualize the optimal transport plan between two structural representations.
        
        Args:
            hybrid1: First hybrid representation
            hybrid2: Second hybrid representation
            title: Title for the visualization
            
        Returns:
            matplotlib.figure.Figure: OT plan visualization
        """
        # Get structural representations
        coef1 = hybrid1['structural_normalized']
        coef2 = hybrid2['structural_normalized']
        
        # Ensure equal length
        max_len = max(len(coef1), len(coef2))
        c1 = np.zeros(max_len)
        c2 = np.zeros(max_len)
        c1[:len(coef1)] = coef1
        c2[:len(coef2)] = coef2
        
        # Normalize to create distributions
        p = c1 / (np.sum(c1) + 1e-10)
        q = c2 / (np.sum(c2) + 1e-10)
        
        # Compute cost matrix
        M = np.abs(np.subtract.outer(np.arange(max_len), np.arange(max_len)))
        M /= M.max() + 1e-10  # Normalize
        
        # Compute OT plan using Sinkhorn
        epsilon = 0.1  # Regularization parameter
        ot_distance, log = ot.sinkhorn2(p, q, M, epsilon, log=True)
        ot_plan = log['pi']
        
        # Create figure for visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot transport plan as a heatmap
        im = ax.imshow(ot_plan, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=ax, label='Transport Weight')
        
        # Set labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Optimal Transport Plan (Distance: {ot_distance:.4f})')
            
        ax.set_xlabel('Second Code Structure Components')
        ax.set_ylabel('First Code Structure Components')
        
        plt.tight_layout()
        
        return fig, ot_distance

# Part 3: Application to Retrieval-Augmented Generation
# -------------------------------------------------

class RAGCodeRetriever:
    """
    Retrieval system for code snippets using hybrid representation.
    Designed to enhance RAG systems for code generation.
    """
    
    def __init__(self, hybrid_rep=None):
        self.hybrid_rep = hybrid_rep or HybridCodeRepresentation()
        self.code_database = []
        
    def add_to_database(self, 
                       code: str, 
                       polynomial_coefficients: np.ndarray, 
                       metadata: Dict[str, Any] = None):
        """
        Add a code snippet to the retrieval database.
        
        Args:
            code: Source code string
            polynomial_coefficients: Structural representation
            metadata: Additional information about the code snippet
        """
        hybrid = self.hybrid_rep.create_hybrid_representation(code, polynomial_coefficients)
        
        entry = {
            'code': code,
            'hybrid': hybrid,
            'metadata': metadata or {}
        }
        
        self.code_database.append(entry)
    
    def add_code_batch(self, 
                      codes: List[str], 
                      coefficients: List[np.ndarray], 
                      metadatas: List[Dict[str, Any]] = None):
        """
        Add multiple code snippets to the database.
        
        Args:
            codes: List of source code strings
            coefficients: List of polynomial coefficients
            metadatas: List of metadata dictionaries (optional)
        """
        if metadatas is None:
            metadatas = [None] * len(codes)
            
        for code, coefs, meta in zip(codes, coefficients, metadatas):
            self.add_to_database(code, coefs, meta)
            
        print(f"Added {len(codes)} code snippets to database. Total size: {len(self.code_database)}")
        
    def retrieve(self, 
                query_code: str, 
                query_coefficients: np.ndarray, 
                alpha: float = 0.5, 
                top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar code snippets for a query.
        
        Args:
            query_code: Source code to match
            query_coefficients: Structural representation of query
            alpha: Weight for structural vs semantic similarity
            top_k: Number of results to return
            
        Returns:
            list: Top k matching code snippets with similarity scores
        """
        query_hybrid = self.hybrid_rep.create_hybrid_representation(
            query_code, query_coefficients, alpha
        )
        
        results = []
        
        for entry in self.code_database:
            distance = self.hybrid_rep.compute_hybrid_distance(
                query_hybrid, entry['hybrid'], alpha
            )
            
            similarity = 1.0 / (1.0 + distance['combined_distance'])
            
            results.append({
                'code': entry['code'],
                'similarity': similarity,
                'structural_similarity': 1.0 / (1.0 + distance['structural_distance']),
                'semantic_similarity': 1.0 / (1.0 + distance['semantic_distance']),
                'metadata': entry['metadata']
            })
        
        # Sort by combined similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    def search_by_structure(self, 
                           query_coefficients: np.ndarray, 
                           top_k: int = 3,
                           threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for code snippets based only on structural similarity.
        
        Args:
            query_coefficients: Structural representation to match
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            list: Top k structurally similar code snippets
        """
        # Normalize query coefficients
        query_norm = query_coefficients / (np.linalg.norm(query_coefficients) + 1e-10)
        
        results = []
        
        for entry in self.code_database:
            # Compute structural distance using OT
            dist = self.hybrid_rep._compute_ot_distance(
                query_norm, entry['hybrid']['structural_normalized']
            )
            
            # Convert to similarity
            similarity = 1.0 / (1.0 + dist)
            
            if similarity >= threshold:
                results.append({
                    'code': entry['code'],
                    'similarity': similarity,
                    'metadata': entry['metadata']
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    def search_by_semantics(self, 
                           query_code: str, 
                           top_k: int = 3,
                           threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for code snippets based only on semantic similarity.
        
        Args:
            query_code: Source code to match semantically
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            list: Top k semantically similar code snippets
        """
        # Generate query embedding
        query_embedding = self.hybrid_rep.embedder.embed_code(query_code)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        results = []
        
        for entry in self.code_database:
            # Compute semantic similarity (1 - cosine distance)
            sim = 1.0 - cosine(query_norm, entry['hybrid']['semantic_normalized'])
            
            if sim >= threshold:
                results.append({
                    'code': entry['code'],
                    'similarity': sim,
                    'metadata': entry['metadata']
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    def save_database(self, filepath: str):
        """
        Save the code database to a file.
        
        Args:
            filepath: Path to save the database
        """
        import pickle
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.code_database, f)
            
        print(f"Saved database with {len(self.code_database)} entries to {filepath}")
    
    def load_database(self, filepath: str):
        """
        Load the code database from a file.
        
        Args:
            filepath: Path to load the database from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self.code_database = pickle.load(f)
            
        print(f"Loaded database with {len(self.code_database)} entries from {filepath}")

# Part 4: Demonstration
# ------------------

def demonstrate_hybrid_representation():
    """Demonstrate the hybrid representation approach."""
    # Sample code snippets
    code_snippets = [
        # Different implementations of the same function
        """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
""",
        """
def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
""",
        # Functionally similar but different naming
        """
def compute_product(numbers):
    result = 1
    for num in numbers:
        result *= num
    return result
""",
        # Structurally similar but semantically different
        """
def count_words(text):
    if not text:
        return 0
    else:
        return len(text.split())
"""
    ]
    
    # Simulate polynomial coefficients
    # In practice, these would come from the dependency tree analysis
    poly_coefficients = [
        np.array([2, 1, 1, 0, 0]),  # Recursive factorial
        np.array([1, 0, 1, 1, 1]),  # Iterative factorial
        np.array([1, 0, 1, 1, 1]),  # Product (structurally similar to iterative factorial)
        np.array([2, 1, 1, 0, 0]),  # Count words (structurally similar to recursive factorial)
    ]
    
    # Create hybrid representations
    hybrid_rep = HybridCodeRepresentation()
    hybrids = []
    
    for code, poly in zip(code_snippets, poly_coefficients):
        hybrid = hybrid_rep.create_hybrid_representation(code, poly)
        hybrids.append(hybrid)
    
    # Compute and display distance matrix
    n = len(hybrids)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = hybrid_rep.compute_hybrid_distance(hybrids[i], hybrids[j])
                distances[i, j] = dist['combined_distance']
    
    print("Hybrid Distance Matrix:")
    print(distances)
    
    # Visualize in distance space
    labels = ["Recursive Factorial", "Iterative Factorial", "Product Function", "Word Counter"]
    hybrid_rep.visualize_distance_space(hybrids, labels)
    
    # Visualize similarity matrix
    hybrid_rep.visualize_hybrid_similarity_matrix(hybrids, labels)
    
    # Visualize OT plan between recursive and iterative factorial
    hybrid_rep.visualize_ot_plan(
        hybrids[0], hybrids[1], 
        "OT Plan: Recursive vs Iterative Factorial"
    )
    
    # Demonstrate RAG retrieval
    retriever = RAGCodeRetriever(hybrid_rep)
    
    # Add code snippets to database
    for idx, (code, poly) in enumerate(zip(code_snippets, poly_coefficients)):
        retriever.add_to_database(code, poly, {'name': f'snippet_{idx}'})
    
    # Query with a new implementation
    query_code = """
def factorial(num):
    # Recursive implementation of factorial
    if num == 0:
        return 1
    return num * factorial(num - 1)
"""
    
    # Simulate query coefficients - similar to recursive factorial
    query_poly = np.array([2, 1, 1, 0, 0])
    
    # Retrieve similar snippets
    results = retriever.retrieve(query_code, query_poly)
    
    print("\nRetrieval Results for Query:")
    print(f"Query: Recursive factorial implementation (slightly different)")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1} - Similarity: {result['similarity']:.4f}")
        print(f"Structural Similarity: {result['structural_similarity']:.4f}")
        print(f"Semantic Similarity: {result['semantic_similarity']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print("Code:")
        print(result['code'])
    
    return {
        'hybrids': hybrids,
        'distances': distances,
        'retrieval_results': results
    }

if __name__ == "__main__":
    demonstrate_hybrid_representation()
