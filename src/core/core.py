"""
Dependency Tree Polynomials with Optimal Transport for Code Representation

This exploratory implementation demonstrates how dependency tree polynomials 
and optimal transport theory could be combined to create representations of 
code structure and measure distances between different code snippets.
"""

import ast
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import ot  # Python Optimal Transport library

# Part 1: Code Parsing and Dependency Extraction
# ---------------------------------------------

class CodeDependencyExtractor(ast.NodeVisitor):
    """Extract dependency relationships from Python code."""
    
    def __init__(self):
        self.dependencies = []
        self.current_node = None
        self.node_types = set()
        self.dependency_types = set()
        
    def visit(self, node):
        """Visit a node and extract dependencies."""
        parent = self.current_node
        self.current_node = node
        
        # Record node type
        node_type = type(node).__name__
        self.node_types.add(node_type)
        
        # Extract relationship between this node and its parent
        if parent:
            parent_type = type(parent).__name__
            dep_type = f"{parent_type}→{node_type}"
            self.dependency_types.add(dep_type)
            self.dependencies.append((parent, node, dep_type))
        
        # Visit all child nodes
        super().generic_visit(node)
        self.current_node = parent
    
    def get_dependency_graph(self):
        """Convert dependencies to a networkx graph."""
        graph = nx.DiGraph()
        for src, dst, dep_type in self.dependencies:
            graph.add_edge(src, dst, type=dep_type)
        return graph

def extract_dependencies(code_str):
    """Parse Python code and extract dependency structure."""
    tree = ast.parse(code_str)
    extractor = CodeDependencyExtractor()
    extractor.visit(tree)
    return extractor.dependencies, extractor.dependency_types, extractor.get_dependency_graph()

# Part 2: Polynomial Representation of Dependency Structures
# --------------------------------------------------------

def compute_polynomial_coefficients(dependencies, dependency_types):
    """
    Convert dependency relationships to polynomial coefficients.
    
    The polynomial has the form:
    P(x₁, x₂, ..., xₙ) = c₁x₁ + c₂x₂ + ... + cₙxₙ
    
    Where:
    - xᵢ represents a dependency type
    - cᵢ is the coefficient (frequency of that dependency)
    """
    # Count occurrences of each dependency type
    dependency_counts = Counter([dep_type for _, _, dep_type in dependencies])
    
    # Create a mapping of dependency types to polynomial variables
    dependency_to_var = {dep_type: i for i, dep_type in enumerate(sorted(dependency_types))}
    
    # Create coefficient vector (initialize with zeros)
    coefficients = np.zeros(len(dependency_types))
    
    # Fill in coefficients based on dependency counts
    for dep_type, count in dependency_counts.items():
        if dep_type in dependency_to_var:  # Check to avoid KeyError
            coefficients[dependency_to_var[dep_type]] = count
            
    return coefficients, dependency_to_var

def polynomial_to_string(coefficients, var_mapping):
    """Convert polynomial coefficients to a readable string."""
    inverse_mapping = {idx: var for var, idx in var_mapping.items()}
    terms = []
    
    for i, coef in enumerate(coefficients):
        if coef > 0:
            var = inverse_mapping[i]
            terms.append(f"{coef}·{var}")
    
    return " + ".join(terms) if terms else "0"

# Part 3: Optimal Transport Distance Calculation
# --------------------------------------------

def compute_ot_distance(coef1, coef2, metric='euclidean'):
    """
    Compute the optimal transport distance between two polynomial representations.
    
    Parameters:
    - coef1, coef2: Coefficient vectors of the polynomials
    - metric: Distance metric to use
    
    Returns:
    - Wasserstein distance between the distributions
    """
    # Ensure coefficients have the same length
    max_len = max(len(coef1), len(coef2))
    c1 = np.zeros(max_len)
    c2 = np.zeros(max_len)
    c1[:len(coef1)] = coef1
    c2[:len(coef2)] = coef2
    
    # Normalize to create distributions
    p = c1 / (np.sum(c1) + 1e-10)  # Adding small epsilon to avoid division by zero
    q = c2 / (np.sum(c2) + 1e-10)
    
    # Compute cost matrix
    if metric == 'euclidean':
        M = ot.dist(np.arange(max_len).reshape(-1, 1), np.arange(max_len).reshape(-1, 1))
        M /= M.max()  # Normalize
    else:
        # For testing, use a simple cost matrix based on index distance
        M = np.abs(np.subtract.outer(np.arange(max_len), np.arange(max_len)))
        M /= M.max()
    
    # Compute Wasserstein distance
    # Using the Sinkhorn algorithm for regularized OT
    epsilon = 0.1  # Regularization parameter
    ot_distance = ot.sinkhorn2(p, q, M, epsilon)
    
    return ot_distance

# Part 4: Visualization and Analysis
# --------------------------------

def visualize_dependency_graph(graph, title):
    """Visualize a dependency graph."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.7)
    
    # Draw labels (simplified to show only node types)
    labels = {node: type(node).__name__ for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=10)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    return plt

def analyze_code_similarity(code1, code2):
    """Analyze the similarity between two code snippets."""
    # Extract dependencies
    deps1, dep_types1, graph1 = extract_dependencies(code1)
    deps2, dep_types2, graph2 = extract_dependencies(code2)
    
    # Combine all dependency types
    all_dep_types = dep_types1.union(dep_types2)
    
    # Compute polynomial coefficients
    coef1, var_mapping1 = compute_polynomial_coefficients(deps1, all_dep_types)
    coef2, var_mapping2 = compute_polynomial_coefficients(deps2, all_dep_types)
    
    # Convert to polynomial strings
    poly1 = polynomial_to_string(coef1, var_mapping1)
    poly2 = polynomial_to_string(coef2, var_mapping2)
    
    # Compute OT distance
    distance = compute_ot_distance(coef1, coef2)
    
    # Visualize dependency graphs
    # (These would be displayed in a notebook environment)
    visualize_dependency_graph(graph1, "Code Snippet 1 Dependencies")
    visualize_dependency_graph(graph2, "Code Snippet 2 Dependencies")
    
    return {
        "polynomial1": poly1,
        "polynomial2": poly2,
        "ot_distance": distance,
        "dependency_types": list(all_dep_types),
        "graph1_nodes": len(graph1.nodes()),
        "graph2_nodes": len(graph2.nodes()),
    }

# Part 5: Demonstration with Example Code Snippets
# ----------------------------------------------

def demonstrate():
    """Demonstrate the approach with example code snippets."""
    # Example 1: Two functionally similar code snippets with different implementations
    code1 = """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
"""

    code2 = """
def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
"""

    results = analyze_code_similarity(code1, code2)
    
    print("Example: Two implementations of factorial function")
    print("\nCode Snippet 1:")
    print(code1)
    print("\nCode Snippet 2:")
    print(code2)
    
    print("\nPolynomial Representation 1:")
    print(results["polynomial1"])
    
    print("\nPolynomial Representation 2:")
    print(results["polynomial2"])
    
    print(f"\nOptimal Transport Distance: {results['ot_distance']}")
    
    print("\nDependency Types:")
    for dep_type in results["dependency_types"]:
        print(f"  - {dep_type}")
    
    print(f"\nGraph 1 Size: {results['graph1_nodes']} nodes")
    print(f"Graph 2 Size: {results['graph2_nodes']} nodes")
    
    # Could add more examples to compare different types of code
    
    return results

# Execute demonstration if run directly
if __name__ == "__main__":
    demonstrate()