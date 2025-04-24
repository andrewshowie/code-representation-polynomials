"""
Multilingual Code Support Module for Dependency Tree Polynomials

This module extends the core implementation to support multiple programming languages,
creating a unified polynomial representation across different languages.
"""

import os
import re
import subprocess
import tempfile
import json
import numpy as np
from tree_sitter import Language, Parser
from collections import defaultdict, Counter
import ot
import matplotlib.pyplot as plt

# Part 1: Tree-sitter Integration for Multi-language Parsing
# ---------------------------------------------------------

class TreeSitterExtractor:
    """Extract dependency relationships from multiple languages using tree-sitter."""
    
    def __init__(self):
        # Path configuration for tree-sitter languages
        # In practice, these would be properly installed
        self.languages = {}
        self.parsers = {}
        
        # Load language support for common languages
        self._initialize_languages()
    
    def _initialize_languages(self):
        """Initialize parsers for supported languages."""
        # Define base directory for language repositories
        base_dir = os.path.abspath(os.path.dirname(__file__))
        build_dir = os.path.join(base_dir, '..', '..', 'build')
        
        # Ensure build directory exists
        os.makedirs(build_dir, exist_ok=True)
        
        # Define language repositories paths
        # In production, you should clone these repositories first
        language_repos = {
            'python': os.path.join(base_dir, '..', '..', 'tree-sitter-python'),
            'javascript': os.path.join(base_dir, '..', '..', 'tree-sitter-javascript'),
            'csharp': os.path.join(base_dir, '..', '..', 'tree-sitter-c-sharp'),
            'java': os.path.join(base_dir, '..', '..', 'tree-sitter-java'),
            'cpp': os.path.join(base_dir, '..', '..', 'tree-sitter-cpp')
        }
        
        # Path to the compiled library
        lib_path = os.path.join(build_dir, 'languages.so')
        
        try:
            # Check if repositories exist, if not, try to use pre-built library
            repos_exist = all(os.path.exists(path) for path in language_repos.values())
            
            if repos_exist:
                # Build language library from repositories
                Language.build_library(
                    lib_path,
                    list(language_repos.values())
                )
            elif os.path.exists(lib_path):
                # If lib already exists but repos don't, just load existing
                pass
            else:
                # If neither repos nor lib exist, use fallback
                print("Language repositories not found and no pre-built library exists.")
                print("Using fallback parsing mode.")
                return
            
            # Load languages from the built library
            self.languages['python'] = Language(lib_path, 'python')
            self.languages['javascript'] = Language(lib_path, 'javascript')
            self.languages['csharp'] = Language(lib_path, 'c_sharp')
            self.languages['java'] = Language(lib_path, 'java')
            self.languages['cpp'] = Language(lib_path, 'cpp')
            
            # Create parsers for each language
            for lang, language in self.languages.items():
                parser = Parser()
                parser.set_language(language)
                self.parsers[lang] = parser
                
            print(f"Successfully initialized parsers for {', '.join(self.languages.keys())}")
                
        except Exception as e:
            print(f"Language initialization error: {e}")
            print("Using fallback parsing mode for demonstration")
    
    def extract_dependencies(self, code, language):
        """
        Extract dependencies from code in the specified language.
        
        Args:
            code (str): The source code to analyze
            language (str): The programming language ('python', 'javascript', etc.)
            
        Returns:
            tuple: (dependencies, dependency_types, nodes)
        """
        # Real implementation would use tree-sitter here
        # For demonstration, we'll use a simplified approach
        if language not in self.parsers:
            return self._fallback_extraction(code, language)
        
        parser = self.parsers[language]
        tree = parser.parse(bytes(code, 'utf8'))
        
        # Process the tree to extract dependencies
        dependencies = []
        dependency_types = set()
        nodes = []
        
        # Traverse the tree and extract relationships
        root_node = tree.root_node
        self._traverse_node(root_node, None, dependencies, dependency_types, nodes)
        
        return dependencies, dependency_types, nodes
    
    def _traverse_node(self, node, parent, dependencies, dependency_types, nodes):
        """Recursively traverse the syntax tree and extract relationships."""
        nodes.append(node)
        
        # Extract relationship with parent
        if parent:
            dep_type = f"{parent.type}→{node.type}"
            dependency_types.add(dep_type)
            dependencies.append((parent, node, dep_type))
        
        # Process children
        for child in node.children:
            self._traverse_node(child, node, dependencies, dependency_types, nodes)
    
    def _fallback_extraction(self, code, language):
        """Fallback extraction method for demonstration purposes."""
        # For demo purposes, we'll use a simple regex-based approach
        dependencies = []
        dependency_types = set()
        nodes = []
        
        # Create a dummy node structure based on the language
        if language == 'python':
            # Extract function definitions
            for match in re.finditer(r'def\s+(\w+)\s*\(', code):
                func_name = match.group(1)
                func_node = {'type': 'function_definition', 'name': func_name}
                nodes.append(func_node)
                
                # Look for function calls
                for call_match in re.finditer(r'(\w+)\s*\(', code):
                    call_name = call_match.group(1)
                    if call_name != func_name:  # Avoid matching the definition
                        call_node = {'type': 'function_call', 'name': call_name}
                        nodes.append(call_node)
                        
                        # Create a dependency
                        dep_type = "function_definition→function_call"
                        dependency_types.add(dep_type)
                        dependencies.append((func_node, call_node, dep_type))
        
        elif language == 'javascript':
            # Extract function definitions (simplified)
            for match in re.finditer(r'function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*\(', code):
                func_name = match.group(1) if match.group(1) else match.group(2)
                func_node = {'type': 'function_definition', 'name': func_name}
                nodes.append(func_node)
        
        elif language == 'csharp':
            # Extract class and method definitions (simplified)
            for match in re.finditer(r'class\s+(\w+)', code):
                class_name = match.group(1)
                class_node = {'type': 'class_definition', 'name': class_name}
                nodes.append(class_node)
                
                # Look for methods
                for method_match in re.finditer(r'(public|private|protected)\s+\w+\s+(\w+)\s*\(', code):
                    method_name = method_match.group(2)
                    method_node = {'type': 'method_definition', 'name': method_name}
                    nodes.append(method_node)
                    
                    # Create a dependency
                    dep_type = "class_definition→method_definition"
                    dependency_types.add(dep_type)
                    dependencies.append((class_node, method_node, dep_type))
        
        return dependencies, dependency_types, nodes

# Part 2: Language-Specific Dependency Mappings
# -------------------------------------------

class DependencyMapper:
    """Map language-specific dependencies to a unified representation."""
    
    def __init__(self):
        # Define mappings from language-specific to unified dependencies
        self.unified_mappings = {
            'python': {
                'FunctionDef→Call': 'FUNCTION→CALL',
                'FunctionDef→Return': 'FUNCTION→RETURN',
                'If→Return': 'CONDITIONAL→RETURN',
                'For→Assign': 'LOOP→ASSIGNMENT',
                'While→Assign': 'LOOP→ASSIGNMENT',
                # Add more mappings as needed
            },
            'javascript': {
                'FunctionDeclaration→CallExpression': 'FUNCTION→CALL',
                'FunctionDeclaration→ReturnStatement': 'FUNCTION→RETURN',
                'IfStatement→ReturnStatement': 'CONDITIONAL→RETURN',
                'ForStatement→AssignmentExpression': 'LOOP→ASSIGNMENT',
                'WhileStatement→AssignmentExpression': 'LOOP→ASSIGNMENT',
                # Add more mappings as needed
            },
            'csharp': {
                'MethodDeclaration→InvocationExpression': 'FUNCTION→CALL',
                'MethodDeclaration→ReturnStatement': 'FUNCTION→RETURN',
                'IfStatement→ReturnStatement': 'CONDITIONAL→RETURN',
                'ForStatement→AssignmentExpression': 'LOOP→ASSIGNMENT',
                'WhileStatement→AssignmentExpression': 'LOOP→ASSIGNMENT',
                # Add more mappings as needed
            }
            # Add more languages as needed
        }
    
    def map_to_unified(self, dep_type, language):
        """Map a language-specific dependency type to the unified representation."""
        if language not in self.unified_mappings:
            return dep_type  # Return as-is if language not supported
        
        mappings = self.unified_mappings[language]
        return mappings.get(dep_type, dep_type)  # Return original if no mapping exists
    
    def map_dependencies(self, dependencies, language):
        """Map all dependencies to the unified representation."""
        unified_deps = []
        
        for src, dst, dep_type in dependencies:
            unified_type = self.map_to_unified(dep_type, language)
            unified_deps.append((src, dst, unified_type))
        
        return unified_deps

# Part 3: Cross-Language Code Comparison
# -----------------------------------

class CrossLanguageComparator:
    """Compare code structure across different programming languages."""
    
    def __init__(self):
        self.extractor = TreeSitterExtractor()
        self.mapper = DependencyMapper()
    
    def extract_unified_representation(self, code, language):
        """Extract a language-agnostic representation of code structure."""
        # Extract language-specific dependencies
        dependencies, dep_types, nodes = self.extractor.extract_dependencies(code, language)
        
        # Map to unified representation
        unified_deps = self.mapper.map_dependencies(dependencies, language)
        unified_types = set(dep_type for _, _, dep_type in unified_deps)
        
        return unified_deps, unified_types, nodes
    
    def compute_polynomial_coefficients(self, unified_deps, unified_types):
        """Convert unified dependencies to polynomial coefficients."""
        # Count occurrences of each unified dependency type
        dependency_counts = Counter([dep_type for _, _, dep_type in unified_deps])
        
        # Create a mapping of dependency types to polynomial variables
        dependency_to_var = {dep_type: i for i, dep_type in enumerate(sorted(unified_types))}
        
        # Create coefficient vector (initialize with zeros)
        coefficients = np.zeros(len(unified_types))
        
        # Fill in coefficients based on dependency counts
        for dep_type, count in dependency_counts.items():
            if dep_type in dependency_to_var:  # Check to avoid KeyError
                coefficients[dependency_to_var[dep_type]] = count
                
        return coefficients, dependency_to_var
    
    def compare_code(self, code1, lang1, code2, lang2):
        """Compare code structure across different languages."""
        # Extract unified representations
        deps1, types1, nodes1 = self.extract_unified_representation(code1, lang1)
        deps2, types2, nodes2 = self.extract_unified_representation(code2, lang2)
        
        # Combine all dependency types
        all_types = types1.union(types2)
        
        # Compute polynomial coefficients
        coef1, mapping1 = self.compute_polynomial_coefficients(deps1, all_types)
        coef2, mapping2 = self.compute_polynomial_coefficients(deps2, all_types)
        
        # Calculate similarity using optimal transport
        # Prepare for OT calculation
        # Normalize coefficient vectors to create proper distributions
        p = coef1 / (np.sum(coef1) + 1e-10)  # Adding epsilon to avoid division by zero
        q = coef2 / (np.sum(coef2) + 1e-10)
        
        # Create cost matrix based on variable positions in the polynomial
        n = len(all_types)
        M = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        M /= M.max()  # Normalize cost matrix
        
        # Compute OT distance using Sinkhorn algorithm
        epsilon = 0.1  # Regularization parameter
        ot_distance = ot.sinkhorn2(p, q, M, epsilon)
        
        # Calculate similarity as inverse of distance
        similarity = 1.0 / (1.0 + ot_distance)
        
        return {
            'coefficients1': coef1,
            'coefficients2': coef2,
            'mapping': mapping1,  # Same as mapping2 since using all_types for both
            'similarity': similarity,
            'ot_distance': ot_distance,
            'node_count1': len(nodes1),
            'node_count2': len(nodes2),
        }
    
    def visualize_cross_language_dependencies(self, 
                                         code1, lang1, 
                                         code2, lang2,
                                         title=None):
        """
        Visualize dependencies across different programming languages.
        
        Args:
            code1: First code snippet
            lang1: Language of first snippet
            code2: Second code snippet
            lang2: Language of second snippet
            title: Optional title for the visualization
            
        Returns:
            matplotlib.figure.Figure: Figure with dependency visualization
        """
        # Extract unified representations
        deps1, types1, nodes1 = self.extract_unified_representation(code1, lang1)
        deps2, types2, nodes2 = self.extract_unified_representation(code2, lang2)
        
        # Count dependency type occurrences
        counts1 = Counter([dep_type for _, _, dep_type in deps1])
        counts2 = Counter([dep_type for _, _, dep_type in deps2])
        
        # Combine all dependency types
        all_types = sorted(types1.union(types2))
        
        # Create data for visualization
        x = np.arange(len(all_types))
        width = 0.35
        
        # Create counts aligned with all_types
        values1 = [counts1.get(dep_type, 0) for dep_type in all_types]
        values2 = [counts2.get(dep_type, 0) for dep_type in all_types]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create bar chart
        rects1 = ax.bar(x - width/2, values1, width, label=f'{lang1} ({len(nodes1)} nodes)')
        rects2 = ax.bar(x + width/2, values2, width, label=f'{lang2} ({len(nodes2)} nodes)')
        
        # Customize chart
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Cross-language Dependency Comparison: {lang1} vs {lang2}')
            
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Dependency Type')
        
        # Truncate long dependency type names
        truncate = lambda s: s[:20] + '...' if len(s) > 20 else s
        ax.set_xticks(x)
        ax.set_xticklabels([truncate(t) for t in all_types], rotation=45, ha='right')
        
        ax.legend()
        
        # Add value labels on bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f'{height}',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
        
        add_labels(rects1)
        add_labels(rects2)
        
        plt.tight_layout()
        
        return fig

    def visualize_ot_plan(self, 
                        code1, lang1, 
                        code2, lang2, 
                        title=None):
        """
        Visualize the optimal transport plan between code in different languages.
        
        Args:
            code1: First code snippet
            lang1: Language of first snippet
            code2: Second code snippet
            lang2: Language of second snippet
            title: Optional title for the visualization
            
        Returns:
            matplotlib.figure.Figure: Visualization of OT plan
        """
        # Extract unified representations
        deps1, types1, nodes1 = self.extract_unified_representation(code1, lang1)
        deps2, types2, nodes2 = self.extract_unified_representation(code2, lang2)
        
        # Combine all dependency types
        all_types = sorted(types1.union(types2))
        
        # Compute polynomial coefficients
        coef1, mapping1 = self.compute_polynomial_coefficients(deps1, all_types)
        coef2, mapping2 = self.compute_polynomial_coefficients(deps2, all_types)
        
        # Normalize coefficient vectors to create proper distributions
        p = coef1 / (np.sum(coef1) + 1e-10)
        q = coef2 / (np.sum(coef2) + 1e-10)
        
        # Create cost matrix based on variable positions
        n = len(all_types)
        M = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        M /= M.max()  # Normalize cost matrix
        
        # Compute OT plan using Sinkhorn algorithm
        epsilon = 0.1  # Regularization parameter
        ot_distance, log = ot.sinkhorn2(p, q, M, epsilon, log=True)
        ot_plan = log['pi']
        
        # Visualize the transport plan
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot transport plan as a heatmap
        im = ax.imshow(ot_plan, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=ax, label='Transport Weight')
        
        # Set labels
        # Truncate long labels for readability
        truncate = lambda s: s[:15] + '...' if len(s) > 15 else s
        type_labels = [truncate(t) for t in all_types]
        
        # Set tick labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(type_labels, rotation=90, ha='right')
        ax.set_yticklabels(type_labels)
        
        # Add title and labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Optimal Transport Plan: {lang1} to {lang2} (Distance: {ot_distance:.4f})')
        
        ax.set_xlabel(f'{lang2} Dependencies')
        ax.set_ylabel(f'{lang1} Dependencies')
        
        plt.tight_layout()
        
        return fig, ot_distance

# Part 4: Demonstration
# ------------------

def demonstrate_cross_language():
    """Demonstrate cross-language code comparison."""
    # Python factorial implementation
    python_code = """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
"""

    # JavaScript factorial implementation
    javascript_code = """
function factorial(n) {
    if (n <= 1) {
        return 1;
    } else {
        return n * factorial(n-1);
    }
}
"""

    # C# factorial implementation
    csharp_code = """
public int Factorial(int n)
{
    if (n <= 1)
    {
        return 1;
    }
    else
    {
        return n * Factorial(n-1);
    }
}
"""

    comparator = CrossLanguageComparator()
    
    # Compare Python and JavaScript
    py_js_comparison = comparator.compare_code(python_code, 'python', javascript_code, 'javascript')
    
    # Compare Python and C#
    py_cs_comparison = comparator.compare_code(python_code, 'python', csharp_code, 'csharp')
    
    # Compare JavaScript and C#
    js_cs_comparison = comparator.compare_code(javascript_code, 'javascript', csharp_code, 'csharp')
    
    print("Cross-language Code Comparison Results")
    print("=====================================")
    
    print(f"\nPython-JavaScript Similarity: {py_js_comparison['similarity']:.4f}")
    print(f"Python-C# Similarity: {py_cs_comparison['similarity']:.4f}")
    print(f"JavaScript-C# Similarity: {js_cs_comparison['similarity']:.4f}")
    
    # Create visualization
    comparator.visualize_cross_language_dependencies(
        python_code, 'python', 
        javascript_code, 'javascript', 
        title='Python vs JavaScript Factorial Function'
    )
    
    # Create OT plan visualization
    fig, distance = comparator.visualize_ot_plan(
        python_code, 'python',
        csharp_code, 'csharp',
        title='Optimal Transport Plan Between Python and C# Implementation'
    )
    
    return {
        'py_js': py_js_comparison,
        'py_cs': py_cs_comparison,
        'js_cs': js_cs_comparison
    }

if __name__ == "__main__":
    demonstrate_cross_language()
