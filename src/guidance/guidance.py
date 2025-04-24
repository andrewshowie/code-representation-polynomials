"""
Structural Guidance for Large Language Models

This module implements a novel approach to guiding LLM code generation
using dependency tree polynomials as structural templates.
"""

import numpy as np
import json
import re
from typing import List, Dict, Tuple, Any, Optional, Union, Set
import ast
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Part 1: Structural Template Extraction
# -----------------------------------

class StructuralTemplate:
    """
    A structural template derived from code using dependency tree polynomials.
    Used to guide LLM generation toward specific structural patterns.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a structural template.
        
        Args:
            name: Template name
            description: Template description
        """
        self.name = name
        self.description = description
        self.dependency_patterns = []
        self.polynomial_coefficients = None
        self.code_examples = []
        
    def add_dependency_pattern(self, src_type: str, dst_type: str, relationship: str):
        """
        Add a dependency pattern to the template.
        
        Args:
            src_type: Source node type
            dst_type: Destination node type
            relationship: Type of relationship
        """
        self.dependency_patterns.append((src_type, dst_type, relationship))
        
    def set_polynomial_coefficients(self, coefficients: np.ndarray):
        """
        Set polynomial coefficients for the template.
        
        Args:
            coefficients: Polynomial coefficient vector
        """
        self.polynomial_coefficients = coefficients
        
    def add_code_example(self, code: str):
        """
        Add a code example that follows this structural template.
        
        Args:
            code: Code string example
        """
        self.code_examples.append(code)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'dependency_patterns': self.dependency_patterns,
            'polynomial_coefficients': self.polynomial_coefficients.tolist() if self.polynomial_coefficients is not None else None,
            'code_examples': self.code_examples
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuralTemplate':
        """
        Create template from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            StructuralTemplate: Reconstructed template
        """
        template = cls(data['name'], data['description'])
        template.dependency_patterns = data['dependency_patterns']
        
        if data['polynomial_coefficients'] is not None:
            template.polynomial_coefficients = np.array(data['polynomial_coefficients'])
            
        template.code_examples = data['code_examples']
        
        return template
    
    @classmethod
    def from_code(cls, code: str, name: str, description: str = "") -> 'StructuralTemplate':
        """
        Create a template from code.
        
        Args:
            code: Code to extract template from
            name: Template name
            description: Template description
            
        Returns:
            StructuralTemplate: Extracted template
        """
        template = cls(name, description)
        
        # Extract AST
        try:
            tree = ast.parse(code)
        except SyntaxError:
            raise ValueError("Invalid Python code")
        
        # Extract dependencies
        extractor = DependencyExtractor()
        extractor.visit(tree)
        
        # Add patterns
        template.dependency_patterns = extractor.dependencies
        
        # Compute polynomial coefficients
        dep_types = set(dep_type for _, _, dep_type in extractor.dependencies)
        counts = Counter(dep_type for _, _, dep_type in extractor.dependencies)
        
        # Create coefficient vector
        coefficients = np.zeros(len(dep_types))
        
        for i, dep_type in enumerate(sorted(dep_types)):
            coefficients[i] = counts[dep_type]
            
        template.set_polynomial_coefficients(coefficients)
        
        # Add the example
        template.add_code_example(code)
        
        return template

class DependencyExtractor(ast.NodeVisitor):
    """Extract dependency relationships from Python code."""
    
    def __init__(self):
        self.dependencies = []
        self.current_node = None
        
    def visit(self, node):
        """Visit a node and extract dependencies."""
        parent = self.current_node
        self.current_node = node
        
        # Extract relationship with parent
        if parent:
            parent_type = type(parent).__name__
            node_type = type(node).__name__
            relationship = f"{parent_type}→{node_type}"
            self.dependencies.append((parent_type, node_type, relationship))
        
        # Visit children
        super().generic_visit(node)
        self.current_node = parent

# Part 2: Template-Based Generation Strategies
# ----------------------------------------

class StructuralPromptStrategy:
    """
    Strategy for enhancing LLM prompts with structural guidance.
    """
    
    def create_structural_prompt(self, 
                                template: StructuralTemplate, 
                                task_description: str,
                                include_examples: bool = True) -> str:
        """
        Create an enhanced prompt with structural guidance.
        
        Args:
            template: Structural template to guide generation
            task_description: Description of the task
            include_examples: Whether to include code examples
            
        Returns:
            str: Enhanced prompt
        """
        # Start with task description
        prompt = f"{task_description}\n\n"
        
        # Add structural guidance
        prompt += "Please implement this following a specific structural pattern:\n\n"
        
        # Add structural description
        prompt += f"Structure: {template.description}\n\n"
        
        # Add key dependency patterns
        prompt += "The code should have these structural elements:\n"
        for src, dst, rel in sorted(template.dependency_patterns):
            prompt += f"- {src} containing {dst}\n"
        
        # Add examples if requested
        if include_examples and template.code_examples:
            prompt += "\nHere's an example following this structure:\n\n"
            prompt += "```python\n"
            prompt += template.code_examples[0]
            prompt += "\n```\n\n"
        
        prompt += "Please implement the solution following this structural pattern."
        
        return prompt

class StructuralFeedbackStrategy:
    """
    Strategy for providing structural feedback on generated code.
    """
    
    def __init__(self):
        self.dependency_extractor = DependencyExtractor()
        
    def extract_dependencies(self, code: str) -> List[Tuple[str, str, str]]:
        """
        Extract dependencies from code.
        
        Args:
            code: Code to analyze
            
        Returns:
            list: Extracted dependencies
        """
        try:
            tree = ast.parse(code)
            extractor = DependencyExtractor()
            extractor.visit(tree)
            return extractor.dependencies
        except SyntaxError:
            return []
        
    def compute_structural_similarity(self, 
                                    template: StructuralTemplate, 
                                    code: str) -> float:
        """
        Compute structural similarity to template.
        
        Args:
            template: Target structural template
            code: Code to evaluate
            
        Returns:
            float: Similarity score (0-1)
        """
        # Extract dependencies
        code_deps = self.extract_dependencies(code)
        
        # Find matching patterns
        template_patterns = set(dep for dep in template.dependency_patterns)
        code_patterns = set(dep for dep in code_deps)
        
        # Calculate Jaccard similarity
        intersection = template_patterns.intersection(code_patterns)
        union = template_patterns.union(code_patterns)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
    
    def generate_feedback(self, 
                         template: StructuralTemplate, 
                         code: str) -> Dict[str, Any]:
        """
        Generate structured feedback on code alignment with template.
        
        Args:
            template: Target structural template
            code: Code to evaluate
            
        Returns:
            dict: Structural feedback
        """
        # Extract dependencies
        code_deps = self.extract_dependencies(code)
        
        # Count patterns
        template_patterns = Counter(dep for dep in template.dependency_patterns)
        code_patterns = Counter(dep for dep in code_deps)
        
        # Find missing and extra patterns
        missing = []
        for pattern, count in template_patterns.items():
            code_count = code_patterns.get(pattern, 0)
            if code_count < count:
                missing.append((pattern, count - code_count))
        
        extra = []
        for pattern, count in code_patterns.items():
            template_count = template_patterns.get(pattern, 0)
            if count > template_count:
                extra.append((pattern, count - template_count))
        
        # Calculate overall similarity
        similarity = self.compute_structural_similarity(template, code)
        
        return {
            'similarity': similarity,
            'missing_patterns': missing,
            'extra_patterns': extra,
            'feedback': self._create_feedback_text(similarity, missing, extra)
        }
    
    def _create_feedback_text(self, 
                            similarity: float, 
                            missing: List[Tuple[Any, int]], 
                            extra: List[Tuple[Any, int]]) -> str:
        """
        Create human-readable feedback text.
        
        Args:
            similarity: Similarity score
            missing: Missing patterns
            extra: Extra patterns
            
        Returns:
            str: Feedback text
        """
        feedback = f"Structural similarity: {similarity:.2f}\n\n"
        
        if similarity >= 0.9:
            feedback += "The code closely follows the target structure.\n"
        elif similarity >= 0.7:
            feedback += "The code generally follows the target structure with some differences.\n"
        else:
            feedback += "The code significantly deviates from the target structure.\n"
        
        if missing:
            feedback += "\nMissing structural elements:\n"
            for (src, dst, rel), count in missing:
                feedback += f"- {src} containing {dst} ({count}x)\n"
        
        if extra:
            feedback += "\nExtra structural elements:\n"
            for (src, dst, rel), count in extra:
                feedback += f"- {src} containing {dst} ({count}x)\n"
        
        return feedback

# Part 3: Template Repository
# ------------------------

class TemplateRepository:
    """
    Repository for storing and retrieving structural templates.
    """
    
    def __init__(self):
        """Initialize repository."""
        self.templates = {}
        
    def add_template(self, template: StructuralTemplate) -> None:
        """
        Add a template to the repository.
        
        Args:
            template: Template to add
        """
        self.templates[template.name] = template
        
    def get_template(self, name: str) -> Optional[StructuralTemplate]:
        """
        Retrieve a template by name.
        
        Args:
            name: Template name
            
        Returns:
            StructuralTemplate or None: Retrieved template
        """
        return self.templates.get(name)
        
    def remove_template(self, name: str) -> bool:
        """
        Remove a template from the repository.
        
        Args:
            name: Template name
            
        Returns:
            bool: Whether the template was removed
        """
        if name in self.templates:
            del self.templates[name]
            return True
        return False
        
    def list_templates(self) -> List[str]:
        """
        List all template names.
        
        Returns:
            list: Template names
        """
        return list(self.templates.keys())
        
    def find_similar_templates(self, 
                              code: str, 
                              threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find templates similar to the given code.
        
        Args:
            code: Code to compare against templates
            threshold: Similarity threshold
            
        Returns:
            list: Similar templates with similarity scores
        """
        feedback_strategy = StructuralFeedbackStrategy()
        
        # Extract structure from code
        try:
            temp_template = StructuralTemplate.from_code(code, "temp")
        except ValueError:
            return []
            
        # Compare with all templates
        similarities = []
        
        for name, template in self.templates.items():
            similarity = feedback_strategy.compute_structural_similarity(template, code)
            if similarity >= threshold:
                similarities.append((name, similarity))
                
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
        
    def save_to_file(self, filename: str) -> None:
        """
        Save repository to file.
        
        Args:
            filename: File to save to
        """
        data = {name: template.to_dict() for name, template in self.templates.items()}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_from_file(self, filename: str) -> None:
        """
        Load repository from file.
        
        Args:
            filename: File to load from
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            
        self.templates = {}
        for name, template_data in data.items():
            self.templates[name] = StructuralTemplate.from_dict(template_data)

# Part 4: Integration with LLM Code Generation
# ----------------------------------------

class StructuralGuidedGeneration:
    """
    Integrate structural guidance with LLM code generation.
    """
    
    def __init__(self, template_repository: TemplateRepository):
        """
        Initialize guided generation.
        
        Args:
            template_repository: Repository of structural templates
        """
        self.repository = template_repository
        self.prompt_strategy = StructuralPromptStrategy()
        self.feedback_strategy = StructuralFeedbackStrategy()
        
    def generate_with_template(self, 
                              llm_client,
                              task_description: str,
                              template_name: str) -> Dict[str, Any]:
        """
        Generate code using a specific template.
        
        Args:
            llm_client: LLM client (e.g., OpenAI API client)
            task_description: Task description
            template_name: Template name
            
        Returns:
            dict: Generation results
        """
        # Get template
        template = self.repository.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
            
        # Create enhanced prompt
        prompt = self.prompt_strategy.create_structural_prompt(
            template, task_description, include_examples=True
        )
        
        # Generate code
        # (In practice, would call the LLM API)
        # For demonstration, we'll simulate a response
        generated_code = self._simulate_llm_generation(task_description, template)
        
        # Evaluate structural alignment
        feedback = self.feedback_strategy.generate_feedback(template, generated_code)
        
        return {
            'prompt': prompt,
            'generated_code': generated_code,
            'structural_feedback': feedback,
            'template': template_name
        }
        
    def generate_with_auto_template(self, 
                                   llm_client,
                                   task_description: str,
                                   seed_code: str = None) -> Dict[str, Any]:
        """
        Generate code with automatically selected template.
        
        Args:
            llm_client: LLM client
            task_description: Task description
            seed_code: Optional seed code to guide template selection
            
        Returns:
            dict: Generation results
        """
        # Select template
        if seed_code:
            # Find similar templates to seed code
            similar_templates = self.repository.find_similar_templates(seed_code)
            if similar_templates:
                template_name = similar_templates[0][0]
                similarity = similar_templates[0][1]
            else:
                # Create a new template from seed code
                template_name = "auto_generated"
                new_template = StructuralTemplate.from_code(
                    seed_code, template_name, "Auto-generated template"
                )
                self.repository.add_template(new_template)
                similarity = 1.0
        else:
            # Select based on task description
            template_name = self._select_template_for_task(task_description)
            similarity = None
            
        # Generate with selected template
        result = self.generate_with_template(llm_client, task_description, template_name)
        
        # Add selection info
        result['template_selection'] = {
            'method': 'seed_code' if seed_code else 'task_description',
            'similarity': similarity
        }
        
        return result
        
    def iterative_refinement(self, 
                            llm_client,
                            task_description: str,
                            template_name: str,
                            max_iterations: int = 3,
                            target_similarity: float = 0.9) -> Dict[str, Any]:
        """
        Iteratively refine code to match structural template.
        
        Args:
            llm_client: LLM client
            task_description: Task description
            template_name: Template name
            max_iterations: Maximum refinement iterations
            target_similarity: Target similarity threshold
            
        Returns:
            dict: Refinement results
        """
        # Get template
        template = self.repository.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
            
        # Initial generation
        result = self.generate_with_template(llm_client, task_description, template_name)
        generations = [result['generated_code']]
        feedbacks = [result['structural_feedback']]
        
        # Check if refinement needed
        current_similarity = result['structural_feedback']['similarity']
        iterations = 0
        
        while current_similarity < target_similarity and iterations < max_iterations:
            iterations += 1
            
            # Create refinement prompt
            previous_code = generations[-1]
            previous_feedback = feedbacks[-1]['feedback']
            
            refinement_prompt = f"{task_description}\n\n"
            refinement_prompt += "Here is my previous attempt:\n\n"
            refinement_prompt += f"```python\n{previous_code}\n```\n\n"
            refinement_prompt += "Structural feedback:\n"
            refinement_prompt += previous_feedback
            refinement_prompt += "\n\nPlease revise the code to better match the target structure."
            
            # Generate refined code
            # (In practice, would call the LLM API)
            refined_code = self._simulate_llm_refinement(
                previous_code, previous_feedback, template
            )
            
            # Evaluate refinement
            feedback = self.feedback_strategy.generate_feedback(template, refined_code)
            current_similarity = feedback['similarity']
            
            # Store results
            generations.append(refined_code)
            feedbacks.append(feedback)
            
        return {
            'generations': generations,
            'feedbacks': feedbacks,
            'iterations': iterations + 1,  # Include initial generation
            'final_similarity': current_similarity,
            'target_achieved': current_similarity >= target_similarity
        }
    
    def _select_template_for_task(self, task_description: str) -> str:
        """
        Select an appropriate template for a task.
        
        Args:
            task_description: Task description
            
        Returns:
            str: Selected template name
        """
        # In practice, would implement sophisticated selection logic
        # For demonstration, use simple keyword matching
        templates = self.repository.list_templates()
        
        if not templates:
            raise ValueError("No templates available")
        
        # Simple keyword matching
        keywords = {
            'factorial': ['factorial', 'recursive', 'multiplication'],
            'sort': ['sort', 'array', 'list', 'order'],
            'search': ['search', 'find', 'locate'],
            'tree': ['tree', 'binary', 'node', 'traverse'],
            'api': ['api', 'endpoint', 'request', 'response']
        }
        
        # Check for matches
        for template, words in keywords.items():
            if template in templates and any(word in task_description.lower() for word in words):
                return template
                
        # Default to first template
        return templates[0]
    
    def _simulate_llm_generation(self, task_description: str, template: StructuralTemplate) -> str:
        """
        Simulate LLM code generation (for demonstration purposes).
        
        Args:
            task_description: Task description
            template: Structural template
            
        Returns:
            str: Generated code
        """
        # Extract keywords from task description for better simulated response
        keywords = task_description.lower()
        
        # Match task to predefined implementations
        if "factorial" in keywords:
            if "iterative" in keywords or "loop" in keywords:
                return """
def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
"""
            else:
                return """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
"""
        elif "fibonacci" in keywords:
            if "iterative" in keywords or "loop" in keywords:
                return """
def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
"""
            else:
                return """
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""
        elif "sort" in keywords:
            if "merge" in keywords:
                return """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""
            else:
                return """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)
"""
        # Default to returning an example from the template if available
        elif template.code_examples:
            # Adjust variable names to make it seem like a new generation
            code = template.code_examples[0]
            # Simple name substitution to make it look different
            replacements = {'factorial': 'compute', 'fibonacci': 'sequence', 
                           'result': 'output', 'n': 'num', 'i': 'index'}
            for old, new in replacements.items():
                code = code.replace(old, new)
            return code
        else:
            # Generic function template as fallback
            return """
def solution(input_data):
    # Default implementation
    result = process_data(input_data)
    return result

def process_data(data):
    # Placeholder processing
    return data
"""
    
    def _simulate_llm_refinement(self, 
                                previous_code: str, 
                                feedback: str, 
                                template: StructuralTemplate) -> str:
        """
        Simulate LLM code refinement (for demonstration purposes).
        
        Args:
            previous_code: Previously generated code
            feedback: Structural feedback
            template: Target template
            
        Returns:
            str: Refined code
        """
        # Parse the feedback to identify missing patterns
        missing_patterns = []
        if "Missing structural elements:" in feedback:
            missing_section = feedback.split("Missing structural elements:")[1]
            if "Extra structural elements:" in missing_section:
                missing_section = missing_section.split("Extra structural elements:")[0]
            
            # Extract missing patterns from feedback
            for line in missing_section.strip().split("\n"):
                if line.startswith("-"):
                    pattern = line.strip("- ").split(" (")[0]
                    missing_patterns.append(pattern)
        
        # Simple refinement based on missing patterns
        refined_code = previous_code
        
        # Check for specific missing patterns and address them
        if any("FunctionDef→If" in pattern for pattern in missing_patterns):
            # Add conditional logic if missing
            if "if " not in refined_code:
                refined_code = refined_code.replace("return ", "if input_value > 0:\n        return ")
                refined_code += "\n    else:\n        return 0"
                
        elif any("FunctionDef→For" in pattern for pattern in missing_patterns):
            # Add loop if missing
            if "for " not in refined_code:
                refined_code = refined_code.replace("return ", "result = 0\n    for i in range(10):\n        result += i\n    return ")
        
        # If can't make specific refinements, use a template example
        if refined_code == previous_code and template.code_examples:
            # Mark as refined version
            refined_code = "# Refined version to better match structural template\n"
            refined_code += template.code_examples[0]
        
        return refined_code

# Part 5: Demonstration
# ------------------

def demonstrate_structural_guidance():
    """Demonstrate structural guidance for LLMs."""
    # Create template repository
    repo = TemplateRepository()
    
    # Create recursive template
    recursive_template = StructuralTemplate(
        "recursive",
        "Recursive function pattern with base case and recursive case"
    )
    
    # Add dependency patterns
    recursive_template.add_dependency_pattern("Module", "FunctionDef", "Module→FunctionDef")
    recursive_template.add_dependency_pattern("FunctionDef", "If", "FunctionDef→If")
    recursive_template.add_dependency_pattern("If", "Return", "If→Return")
    recursive_template.add_dependency_pattern("If", "Return", "If→Return")
    recursive_template.add_dependency_pattern("Return", "Call", "Return→Call")
    
    # Set coefficients
    recursive_template.set_polynomial_coefficients(np.array([1, 1, 2, 1, 1]))
    
    # Add example
    recursive_template.add_code_example("""
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
""")
    
    # Create iterative template
    iterative_template = StructuralTemplate(
        "iterative",
        "Iterative function pattern with loop and accumulator"
    )
    
    # Add dependency patterns
    iterative_template.add_dependency_pattern("Module", "FunctionDef", "Module→FunctionDef")
    iterative_template.add_dependency_pattern("FunctionDef", "Assign", "FunctionDef→Assign")
    iterative_template.add_dependency_pattern("FunctionDef", "For", "FunctionDef→For")
    iterative_template.add_dependency_pattern("For", "Assign", "For→Assign")
    iterative_template.add_dependency_pattern("FunctionDef", "Return", "FunctionDef→Return")
    
    # Set coefficients
    iterative_template.set_polynomial_coefficients(np.array([1, 1, 1, 1, 1]))
    
    # Add example
    iterative_template.add_code_example("""
def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
""")
    
    # Add templates to repository
    repo.add_template(recursive_template)
    repo.add_template(iterative_template)
    
    # Create guidance system
    guidance = StructuralGuidedGeneration(repo)
    
    # Demonstrate different generation approaches
    
    # 1. Generate with specific template
    print("Generating with specific template (recursive)...")
    result1 = guidance.generate_with_template(None, "Implement a factorial function", "recursive")
    
    # 2. Generate with automatic template selection
    print("\nGenerating with automatic template selection...")
    seed_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    result2 = guidance.generate_with_auto_template(None, "Implement a fibonacci function", seed_code)
    
    # 3. Iterative refinement
    print("\nDemonstrating iterative refinement...")
    result3 = guidance.iterative_refinement(
        None, "Implement a factorial function", "recursive", max_iterations=2
    )
    
    # Print results
    print("\n=== Results ===\n")
    
    print("1. Specific Template Generation:")
    print(f"Template: {result1['template']}")
    print(f"Structural Similarity: {result1['structural_feedback']['similarity']:.4f}")
    print("\nGenerated Code:")
    print(result1['generated_code'])
    print("\nStructural Feedback:")
    print(result1['structural_feedback']['feedback'])
    
    print("\n2. Automatic Template Selection:")
    print(f"Selected Template: {result2['template']}")
    print(f"Selection Method: {result2['template_selection']['method']}")
    if result2['template_selection']['similarity'] is not None:
        print(f"Similarity to Seed: {result2['template_selection']['similarity']:.4f}")
    print("\nGenerated Code:")
    print(result2['generated_code'])
    
    print("\n3. Iterative Refinement:")
    print(f"Iterations: {result3['iterations']}")
    print(f"Final Similarity: {result3['final_similarity']:.4f}")
    print(f"Target Achieved: {result3['target_achieved']}")
    print("\nFinal Generation:")
    print(result3['generations'][-1])
    
    # Visualize similarity progression
    plt.figure(figsize=(10, 6))
    iterations = range(result3['iterations'])
    similarities = [result3['feedbacks'][i]['similarity'] for i in range(result3['iterations'])]
    
    plt.plot(iterations, similarities, marker='o', linestyle='-', linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target Similarity')
    
    plt.xlabel('Iteration')
    plt.ylabel('Structural Similarity')
    plt.title('Structural Similarity Progression During Refinement')
    plt.grid(True)
    plt.legend()
    
    return {
        'specific_template': result1,
        'auto_template': result2,
        'iterative_refinement': result3
    }

if __name__ == "__main__":
    demonstrate_structural_guidance()
