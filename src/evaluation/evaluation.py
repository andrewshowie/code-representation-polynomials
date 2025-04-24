"""
Evaluation Framework for Code Generation

This module implements an evaluation framework for assessing code generation quality
based on dependency tree polynomials and optimal transport. It provides metrics
that account for structural fidelity beyond simple string matching.
"""

import numpy as np
import json
import dataclasses
from typing import List, Dict, Tuple, Any, Optional, Union, Set
from collections import defaultdict
import ot
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Part 1: Code Generation Metrics
# -----------------------------

@dataclasses.dataclass
class CodeGenerationMetrics:
    """Container for code generation evaluation metrics."""
    exact_match: float = 0.0
    bleu_score: float = 0.0
    structural_similarity: float = 0.0
    semantic_similarity: float = 0.0
    hybrid_similarity: float = 0.0
    
    # Specific structural metrics
    dependency_precision: float = 0.0
    dependency_recall: float = 0.0
    dependency_f1: float = 0.0
    
    # OT-specific metrics
    ot_distance: float = 0.0
    ot_plan_entropy: float = 0.0
    ot_coverage: float = 0.0
    flow_concentration: float = 0.0
    structural_deformation: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return dataclasses.asdict(self)
    
    def __str__(self) -> str:
        """String representation of metrics."""
        parts = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            parts.append(f"{field.name}: {value:.4f}")
        return "\n".join(parts)

class CodeEvaluator:
    """Evaluate code generation using polynomial representation and optimal transport."""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize evaluator.
        
        Args:
            alpha: Weight balancing structural vs semantic metrics
        """
        self.alpha = alpha
    
    def compute_exact_match(self, reference: str, generated: str) -> float:
        """Compute exact match score."""
        return 1.0 if reference.strip() == generated.strip() else 0.0
    
    def compute_bleu(self, reference: str, generated: str) -> float:
        """
        Compute BLEU score using NLTK for accuracy.
        
        Args:
            reference: Reference code string
            generated: Generated code string
            
        Returns:
            float: BLEU score (0-1)
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Tokenize into words
            ref_tokens = reference.strip().split()
            gen_tokens = generated.strip().split()
            
            # Handle empty sequences
            if not gen_tokens:
                return 0.0
            if not ref_tokens:
                return 0.0
            
            # Use smoothing to handle edge cases (such as n-gram precision being 0)
            smoothing = SmoothingFunction().method1
            
            # Calculate BLEU with equal weights for 1-4 grams
            weights = (0.25, 0.25, 0.25, 0.25)
            score = sentence_bleu([ref_tokens], gen_tokens, weights=weights, 
                                 smoothing_function=smoothing)
            
            return score
            
        except ImportError:
            # Fall back to simplified implementation if NLTK isn't available
            # Tokenize
            ref_tokens = reference.strip().split()
            gen_tokens = generated.strip().split()
            
            # Count matches (1-gram)
            matches = sum(1 for token in gen_tokens if token in ref_tokens)
            
            # Precision component
            precision = matches / len(gen_tokens) if gen_tokens else 0
            
            # Apply brevity penalty
            bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(gen_tokens), 1)))
            
            return bp * precision
    
    def compute_structural_metrics(self, 
                                  ref_dependencies: List[Tuple], 
                                  gen_dependencies: List[Tuple]) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 for dependency structures.
        
        Args:
            ref_dependencies: Reference code dependencies
            gen_dependencies: Generated code dependencies
            
        Returns:
            dict: Precision, recall, and F1 scores
        """
        # Extract dependency types
        ref_dep_types = set(dep_type for _, _, dep_type in ref_dependencies)
        gen_dep_types = set(dep_type for _, _, dep_type in gen_dependencies)
        
        # Count matches
        matches = ref_dep_types.intersection(gen_dep_types)
        
        # Calculate metrics
        precision = len(matches) / len(gen_dep_types) if gen_dep_types else 0.0
        recall = len(matches) / len(ref_dep_types) if ref_dep_types else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def compute_ot_metrics(self, 
                          ref_coefficients: np.ndarray, 
                          gen_coefficients: np.ndarray) -> Dict[str, float]:
        """
        Compute optimal transport metrics between polynomial coefficients.
        
        Args:
            ref_coefficients: Reference code polynomial coefficients
            gen_coefficients: Generated code polynomial coefficients
            
        Returns:
            dict: OT distance, plan entropy, coverage, and flow concentration
        """
        # Ensure equal length
        max_len = max(len(ref_coefficients), len(gen_coefficients))
        ref_coef = np.zeros(max_len)
        gen_coef = np.zeros(max_len)
        ref_coef[:len(ref_coefficients)] = ref_coefficients
        gen_coef[:len(gen_coefficients)] = gen_coefficients
        
        # Normalize to create distributions
        p = ref_coef / (np.sum(ref_coef) + 1e-10)
        q = gen_coef / (np.sum(gen_coef) + 1e-10)
        
        # Compute cost matrix
        M = np.abs(np.subtract.outer(np.arange(max_len), np.arange(max_len)))
        M /= M.max()  # Normalize
        
        # Compute OT plan using Sinkhorn algorithm
        epsilon = 0.1  # Regularization parameter
        ot_distance, log = ot.sinkhorn2(p, q, M, epsilon, log=True)
        ot_plan = log['pi']
        
        # Calculate plan entropy
        plan_entropy = -np.sum(ot_plan * np.log(ot_plan + 1e-10))
        
        # Calculate coverage (how much of reference is covered)
        coverage = np.sum(np.any(ot_plan > 0.01, axis=1))
        coverage_ratio = coverage / max(len(ref_coefficients), 1)
        
        # Calculate flow concentration (Gini coefficient of the OT plan)
        # Flatten the OT plan and compute Gini coefficient
        flat_plan = ot_plan.flatten()
        sorted_plan = np.sort(flat_plan)
        n = len(sorted_plan)
        cumsum = np.cumsum(sorted_plan)
        flow_concentration = (n + 1 - 2 * np.sum(cumsum) / (cumsum[-1] * n + 1e-10)) / max(n, 1)
        
        # Calculate structural deformation (scaled total transport cost)
        deformation = ot_distance / (np.sum(p) * np.sum(q) + 1e-10)
        
        return {
            'ot_distance': ot_distance,
            'ot_plan_entropy': plan_entropy,
            'ot_coverage': coverage_ratio,
            'flow_concentration': flow_concentration,
            'structural_deformation': deformation
        }
    
    def evaluate(self, 
                reference: str, 
                generated: str,
                ref_dependencies: List[Tuple],
                gen_dependencies: List[Tuple],
                ref_coefficients: np.ndarray,
                gen_coefficients: np.ndarray,
                semantic_similarity: float) -> CodeGenerationMetrics:
        """
        Evaluate generated code against reference.
        
        Args:
            reference: Reference code string
            generated: Generated code string
            ref_dependencies: Reference code dependencies
            gen_dependencies: Generated code dependencies
            ref_coefficients: Reference code polynomial coefficients
            gen_coefficients: Generated code polynomial coefficients
            semantic_similarity: Semantic similarity score from embeddings
            
        Returns:
            CodeGenerationMetrics: Evaluation metrics
        """
        # String-based metrics
        exact_match = self.compute_exact_match(reference, generated)
        bleu = self.compute_bleu(reference, generated)
        
        # Structural metrics
        struct_metrics = self.compute_structural_metrics(ref_dependencies, gen_dependencies)
        
        # OT metrics
        ot_metrics = self.compute_ot_metrics(ref_coefficients, gen_coefficients)
        
        # Compute structural similarity based on OT distance
        structural_similarity = 1.0 / (1.0 + ot_metrics['ot_distance'])
        
        # Compute hybrid similarity
        hybrid_similarity = self.alpha * structural_similarity + (1 - self.alpha) * semantic_similarity
        
        # Create metrics object
        metrics = CodeGenerationMetrics(
            exact_match=exact_match,
            bleu_score=bleu,
            structural_similarity=structural_similarity,
            semantic_similarity=semantic_similarity,
            hybrid_similarity=hybrid_similarity,
            dependency_precision=struct_metrics['precision'],
            dependency_recall=struct_metrics['recall'],
            dependency_f1=struct_metrics['f1'],
            ot_distance=ot_metrics['ot_distance'],
            ot_plan_entropy=ot_metrics['ot_plan_entropy'],
            ot_coverage=ot_metrics['ot_coverage'],
            flow_concentration=ot_metrics.get('flow_concentration', 0.0),
            structural_deformation=ot_metrics.get('structural_deformation', 0.0)
        )
        
        return metrics
    
    def visualize_ot_plan(self, 
                         ref_coefficients: np.ndarray, 
                         gen_coefficients: np.ndarray,
                         ref_dependencies: List[Tuple],
                         gen_dependencies: List[Tuple]) -> plt.Figure:
        """
        Visualize the optimal transport plan between two code structures.
        
        Args:
            ref_coefficients: Reference code polynomial coefficients
            gen_coefficients: Generated code polynomial coefficients
            ref_dependencies: Reference dependencies for labels
            gen_dependencies: Generated dependencies for labels
            
        Returns:
            matplotlib.figure.Figure: Figure with OT plan visualization
        """
        # Ensure equal length
        max_len = max(len(ref_coefficients), len(gen_coefficients))
        ref_coef = np.zeros(max_len)
        gen_coef = np.zeros(max_len)
        ref_coef[:len(ref_coefficients)] = ref_coefficients
        gen_coef[:len(gen_coefficients)] = gen_coefficients
        
        # Normalize to create distributions
        p = ref_coef / (np.sum(ref_coef) + 1e-10)
        q = gen_coef / (np.sum(gen_coef) + 1e-10)
        
        # Compute cost matrix
        M = np.abs(np.subtract.outer(np.arange(max_len), np.arange(max_len)))
        M /= M.max()  # Normalize
        
        # Compute OT plan using Sinkhorn algorithm
        epsilon = 0.1  # Regularization parameter
        ot_distance, log = ot.sinkhorn2(p, q, M, epsilon, log=True)
        ot_plan = log['pi']
        
        # Create figure for visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot transport plan as a heatmap
        im = ax.imshow(ot_plan, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=ax, label='Transport Weight')
        
        # Set labels
        # Extract dependency types from tuples
        ref_dep_types = [dep_type for _, _, dep_type in ref_dependencies[:max_len]]
        gen_dep_types = [dep_type for _, _, dep_type in gen_dependencies[:max_len]]
        
        # Truncate long labels for readability
        truncate = lambda s: s[:15] + '...' if len(s) > 15 else s
        ref_labels = [truncate(t) for t in ref_dep_types[:max_len]]
        gen_labels = [truncate(t) for t in gen_dep_types[:max_len]]
        
        # Set tick labels
        ax.set_xticks(np.arange(len(gen_labels)))
        ax.set_yticks(np.arange(len(ref_labels)))
        ax.set_xticklabels(gen_labels, rotation=45, ha='right')
        ax.set_yticklabels(ref_labels)
        
        # Add title and labels
        ax.set_title(f'Optimal Transport Plan (Distance: {ot_distance:.4f})')
        ax.set_xlabel('Generated Code Dependencies')
        ax.set_ylabel('Reference Code Dependencies')
        
        plt.tight_layout()
        
        return fig

# Part 2: Test Suite Management
# --------------------------

class TestCase:
    """Test case for code generation evaluation."""
    
    def __init__(self, 
                description: str,
                reference_code: str,
                reference_dependencies: List[Tuple],
                reference_coefficients: np.ndarray,
                reference_semantic: Optional[np.ndarray] = None):
        """
        Initialize test case.
        
        Args:
            description: Test case description
            reference_code: Reference code string
            reference_dependencies: Reference dependency structure
            reference_coefficients: Reference polynomial coefficients
            reference_semantic: Reference semantic embedding (optional)
        """
        self.description = description
        self.reference_code = reference_code
        self.reference_dependencies = reference_dependencies
        self.reference_coefficients = reference_coefficients
        self.reference_semantic = reference_semantic

class TestSuite:
    """Test suite for evaluating code generation models."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_cases = []
        
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
        
    def load_from_file(self, filename: str) -> None:
        """
        Load test cases from file.
        
        Args:
            filename: Path to the test cases file (JSON format)
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.test_cases = []
            
            for tc_data in data:
                test_case = TestCase(
                    description=tc_data['description'],
                    reference_code=tc_data['reference_code'],
                    reference_dependencies=tc_data.get('reference_dependencies', []),
                    reference_coefficients=np.array(tc_data.get('reference_coefficients', [])),
                    reference_semantic=np.array(tc_data.get('reference_semantic', None)) 
                        if tc_data.get('reference_semantic') else None
                )
                self.test_cases.append(test_case)
                
            print(f"Loaded {len(self.test_cases)} test cases from {filename}")
            
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading test cases from {filename}: {e}")
        
    def save_to_file(self, filename: str) -> None:
        """
        Save test cases to file.
        
        Args:
            filename: Path to save the test cases (JSON format)
        """
        try:
            data = []
            
            for tc in self.test_cases:
                tc_data = {
                    'description': tc.description,
                    'reference_code': tc.reference_code,
                    'reference_dependencies': tc.reference_dependencies,
                    'reference_coefficients': tc.reference_coefficients.tolist(),
                }
                
                if tc.reference_semantic is not None:
                    tc_data['reference_semantic'] = tc.reference_semantic.tolist()
                    
                data.append(tc_data)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Saved {len(data)} test cases to {filename}")
            
        except IOError as e:
            print(f"Error saving test cases to {filename}: {e}")
        
    def evaluate_model(self, 
                      model_fn,
                      dependency_extractor,
                      coefficient_calculator,
                      semantic_calculator) -> Dict[str, Any]:
        """
        Evaluate a model on all test cases.
        
        Args:
            model_fn: Function that takes a description and returns generated code
            dependency_extractor: Function to extract dependencies from code
            coefficient_calculator: Function to calculate polynomial coefficients
            semantic_calculator: Function to calculate semantic similarity
            
        Returns:
            dict: Evaluation results for all test cases
        """
        evaluator = CodeEvaluator()
        results = []
        
        for test_case in self.test_cases:
            # Generate code using the model
            generated_code = model_fn(test_case.description)
            
            # Extract dependencies and calculate coefficients for generated code
            gen_dependencies, gen_dep_types = dependency_extractor(generated_code)
            gen_coefficients = coefficient_calculator(gen_dependencies, gen_dep_types)
            
            # Calculate semantic similarity
            semantic_similarity = semantic_calculator(test_case.reference_code, generated_code)
            
            # Evaluate
            metrics = evaluator.evaluate(
                test_case.reference_code,
                generated_code,
                test_case.reference_dependencies,
                gen_dependencies,
                test_case.reference_coefficients,
                gen_coefficients,
                semantic_similarity
            )
            
            # Store results
            result = {
                'test_case': test_case.description,
                'reference_code': test_case.reference_code,
                'generated_code': generated_code,
                'metrics': metrics.to_dict()
            }
            
            results.append(result)
            
        # Aggregate results
        aggregate_metrics = self._aggregate_metrics(results)
        
        return {
            'test_cases': results,
            'aggregate': aggregate_metrics
        }
    
    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across all test cases."""
        # Extract metrics
        metrics_list = [result['metrics'] for result in results]
        
        # Calculate averages
        aggregate = {}
        if metrics_list:
            for key in metrics_list[0]:
                aggregate[key] = np.mean([m[key] for m in metrics_list])
                
        return aggregate
        
    def visualize_results(self, results: Dict[str, Any]) -> None:
        """Visualize evaluation results."""
        # Extract results
        test_cases = results['test_cases']
        aggregate = results['aggregate']
        
        # Create bar chart for aggregate metrics
        plt.figure(figsize=(12, 6))
        
        # Select key metrics to visualize
        key_metrics = [
            'exact_match', 
            'bleu_score', 
            'structural_similarity', 
            'semantic_similarity', 
            'hybrid_similarity', 
            'dependency_f1'
        ]
        
        values = [aggregate[key] for key in key_metrics]
        
        plt.bar(key_metrics, values)
        plt.ylim(0, 1.0)
        plt.title('Aggregate Code Generation Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        
        # Create second plot for individual test cases
        plt.figure(figsize=(12, 8))
        
        # Extract hybrid similarity for each test case
        case_descriptions = [case['test_case'][:20] + '...' for case in test_cases]
        hybrid_scores = [case['metrics']['hybrid_similarity'] for case in test_cases]
        structural_scores = [case['metrics']['structural_similarity'] for case in test_cases]
        semantic_scores = [case['metrics']['semantic_similarity'] for case in test_cases]
        
        # Plot
        x = np.arange(len(case_descriptions))
        width = 0.25
        
        plt.bar(x - width, structural_scores, width, label='Structural')
        plt.bar(x, semantic_scores, width, label='Semantic')
        plt.bar(x + width, hybrid_scores, width, label='Hybrid')
        
        plt.ylabel('Similarity Score')
        plt.title('Similarity Scores by Test Case')
        plt.xticks(x, case_descriptions, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        return plt

# Part 3: Demonstration
# ------------------

def demonstrate_evaluation():
    """Demonstrate the evaluation framework."""
    # Create simple test cases
    test_cases = [
        {
            'description': 'Implement a factorial function',
            'reference_code': """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
""",
            'reference_dependencies': [
                (None, 'FunctionDef', 'Module→FunctionDef'),
                ('FunctionDef', 'If', 'FunctionDef→If'),
                ('If', 'Return', 'If→Return'),
                ('If', 'Return', 'If→Return'),
                ('Return', 'BinOp', 'Return→BinOp')
            ],
            'reference_coefficients': np.array([1, 1, 2, 1, 1])
        },
        {
            'description': 'Implement a function to check if a number is prime',
            'reference_code': """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
""",
            'reference_dependencies': [
                (None, 'FunctionDef', 'Module→FunctionDef'),
                ('FunctionDef', 'If', 'FunctionDef→If'),
                ('If', 'Return', 'If→Return'),
                ('FunctionDef', 'For', 'FunctionDef→For'),
                ('For', 'If', 'For→If'),
                ('If', 'Return', 'If→Return'),
                ('FunctionDef', 'Return', 'FunctionDef→Return')
            ],
            'reference_coefficients': np.array([1, 2, 3, 1, 1, 0, 0])
        }
    ]
    
    # Create test suite
    suite = TestSuite()
    
    # Add test cases to suite
    for tc in test_cases:
        test_case = TestCase(
            tc['description'],
            tc['reference_code'],
            tc['reference_dependencies'],
            tc['reference_coefficients']
        )
        suite.add_test_case(test_case)
    
    # Define mock model and extractors for demonstration
    def mock_model(description):
        """Mock model that generates code based on description."""
        if 'factorial' in description.lower():
            # Generate almost correct factorial but with a small error
            return """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)  # Added space in n-1
"""
        elif 'prime' in description.lower():
            # Generate a less optimal implementation
            return """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):  # Less efficient upper bound
        if n % i == 0:
            return False
    return True
"""
        else:
            return "# Not implemented"
    
    def mock_dependency_extractor(code):
        """Mock dependency extractor."""
        # Very simplified extraction
        dependencies = []
        dependency_types = set()
        
        if 'factorial' in code:
            # Simplified dependencies for factorial
            dependencies = [
                (None, 'FunctionDef', 'Module→FunctionDef'),
                ('FunctionDef', 'If', 'FunctionDef→If'),
                ('If', 'Return', 'If→Return'),
                ('If', 'Return', 'If→Return'),
                ('Return', 'BinOp', 'Return→BinOp')
            ]
        elif 'is_prime' in code:
            # Simplified dependencies for is_prime
            if 'range(2, n)' in code:
                # Less efficient version
                dependencies = [
                    (None, 'FunctionDef', 'Module→FunctionDef'),
                    ('FunctionDef', 'If', 'FunctionDef→If'),
                    ('If', 'Return', 'If→Return'),
                    ('FunctionDef', 'For', 'FunctionDef→For'),
                    ('For', 'If', 'For→If'),
                    ('If', 'Return', 'If→Return'),
                    ('FunctionDef', 'Return', 'FunctionDef→Return')
                ]
            else:
                # Efficient version
                dependencies = [
                    (None, 'FunctionDef', 'Module→FunctionDef'),
                    ('FunctionDef', 'If', 'FunctionDef→If'),
                    ('If', 'Return', 'If→Return'),
                    ('FunctionDef', 'For', 'FunctionDef→For'),
                    ('For', 'If', 'For→If'),
                    ('If', 'Return', 'If→Return'),
                    ('FunctionDef', 'Return', 'FunctionDef→Return')
                ]
        
        # Extract dependency types
        dependency_types = set(dep_type for _, _, dep_type in dependencies)
        
        return dependencies, dependency_types
    
    def mock_coefficient_calculator(dependencies, dep_types):
        """Mock coefficient calculator."""
        # Very simplified coefficient calculation
        if any('factorial' in str(dep) for dep in dependencies):
            return np.array([1, 1, 2, 1, 1])
        elif any('is_prime' in str(dep) for dep in dependencies):
            if any('range(2, n)' in str(dep) for dep in dependencies):
                # Less efficient version
                return np.array([1, 2, 3, 1, 1, 0, 0])
            else:
                # Efficient version
                return np.array([1, 2, 3, 1, 1, 0, 0])
        else:
            return np.array([])
    
    def mock_semantic_calculator(ref_code, gen_code):
        """Mock semantic similarity calculator."""
        # Very simplified similarity calculation
        # In practice, would use embeddings
        if 'factorial' in ref_code and 'factorial' in gen_code:
            return 0.95  # High similarity for factorial
        elif 'is_prime' in ref_code and 'is_prime' in gen_code:
            if 'range(2, n)' in gen_code:
                return 0.85  # Lower similarity for less efficient prime
            else:
                return 0.98  # High similarity for efficient prime
        else:
            return 0.0
    
    # Evaluate model
    results = suite.evaluate_model(
        mock_model,
        mock_dependency_extractor,
        mock_coefficient_calculator,
        mock_semantic_calculator
    )
    
    # Print results
    print("Evaluation Results:")
    print("------------------")
    
    print("\nAggregate Metrics:")
    for key, value in results['aggregate'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTest Case Results:")
    for i, test_case in enumerate(results['test_cases']):
        print(f"\n  Test Case {i+1}: {test_case['test_case']}")
        for key, value in test_case['metrics'].items():
            print(f"    {key}: {value:.4f}")
    
    # Visualize results
    suite.visualize_results(results)
    
    # Visualize OT plan for factorial example
    evaluator = CodeEvaluator()
    factorial_case = results['test_cases'][0]
    ref_deps = test_cases[0]['reference_dependencies']
    gen_deps = mock_dependency_extractor(factorial_case['generated_code'])[0]
    ref_coef = test_cases[0]['reference_coefficients']
    gen_coef = mock_coefficient_calculator(gen_deps, set())
    
    evaluator.visualize_ot_plan(ref_coef, gen_coef, ref_deps, gen_deps)
    
    return results

if __name__ == "__main__":
    demonstrate_evaluation()
