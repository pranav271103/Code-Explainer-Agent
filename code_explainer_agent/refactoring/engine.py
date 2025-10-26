"""
Refactoring engine for the Code Explainer Agent.
"""
from typing import Dict, List, Any, Optional, Tuple, Union, Type, cast
import ast
import astor
from dataclasses import dataclass
from enum import Enum, auto
import logging

from ..core.config import RefactoringConfig

# Try to import astor, but make it optional
try:
    import astor
    HAS_ASTOR = True
except ImportError:
    HAS_ASTOR = False

# Type aliases
ASTNode = Union[
    ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef,
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Expr,
    ast.Pass, ast.Break, ast.Continue, ast.Raise, ast.Return
]


class RefactoringType(Enum):
    """Types of refactoring operations."""
    SPLIT_FUNCTION = "split_function"
    RENAME_VARIABLE = "rename_variable"
    EXTRACT_METHOD = "extract_method"
    REMOVE_UNUSED_IMPORTS = "remove_unused_imports"
    SIMPLIFY_CONDITIONAL = "simplify_conditional"
    REMOVE_DUPLICATE_CODE = "remove_duplicate_code"
    ADD_TYPE_HINTS = "add_type_hints"
    IMPROVE_VARIABLE_NAMES = "improve_variable_names"
    FIX_CODE_STYLE = "fix_code_style"
    CONVERT_TO_F_STRING = "convert_to_f_string"


@dataclass
class RefactoringSuggestion:
    """A single refactoring suggestion."""
    refactoring_type: RefactoringType
    description: str
    location: Tuple[int, int]  # (start_line, end_line)
    code_before: str
    code_after: Optional[str] = None
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the suggestion to a dictionary."""
        return {
            'refactoring_type': self.refactoring_type.value,
            'description': self.description,
            'location': self.location,
            'code_before': self.code_before,
            'code_after': self.code_after,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefactoringSuggestion':
        """Create a suggestion from a dictionary."""
        return cls(
            refactoring_type=RefactoringType(data['refactoring_type']),
            description=data['description'],
            location=tuple(data['location']),
            code_before=data['code_before'],
            code_after=data.get('code_after'),
            confidence=data.get('confidence', 0.0),
            metadata=data.get('metadata')
        )


class RefactoringEngine:
    """
    Analyzes code and suggests refactoring improvements.
    """
    
    def __init__(self, config: RefactoringConfig):
        """Initialize the refactoring engine.
        
        Args:
            config: Refactoring configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, ast_data: Dict[str, Any], file_path: str) -> List[RefactoringSuggestion]:
        """Analyze the code and suggest refactorings.
        
        Args:
            ast_data: The parsed AST of the code.
            file_path: Path to the source file.
            
        Returns:
            List of refactoring suggestions.
        """
        suggestions: List[RefactoringSuggestion] = []
        
        try:
            # Only proceed if we have valid AST data
            if not ast_data or 'type' not in ast_data:
                return suggestions
            
            # Convert AST dict to actual AST nodes if needed
            ast_node = self._dict_to_ast(ast_data)
            
            # Check for different types of refactoring opportunities
            if self.config.auto_apply:
                self.logger.info(f"Auto-applying refactorings for {file_path}")
                
            if self.config.split_long_functions:
                self._check_long_functions(ast_node, suggestions, file_path)
                
            if self.config.remove_unused_variables:
                self._check_unused_variables(ast_node, suggestions, file_path)
                
            if self.config.remove_duplicate_code:
                self._check_duplicate_code(ast_node, suggestions, file_path)
            
            # Only include suggestions that meet our confidence threshold
            return [s for s in suggestions if s.confidence >= 0.7]
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {str(e)}", exc_info=True)
            return []
    
    def _dict_to_ast(self, node_dict: Dict[str, Any]) -> ASTNode:
        """Convert an AST dictionary to actual AST nodes."""
        if not isinstance(node_dict, dict) or 'type' not in node_dict:
            return node_dict
            
        node_type = node_dict['type']
        node_class = getattr(ast, node_type, None)
        
        if not node_class:
            self.logger.warning(f"Unknown AST node type: {node_type}")
            return node_dict
            
        # Create the node with its attributes
        node = node_class()
        
        # Set attributes
        for key, value in node_dict.items():
            if key == 'type':
                continue
                
            if isinstance(value, dict):
                setattr(node, key, self._dict_to_ast(value))
            elif isinstance(value, list):
                setattr(node, key, [self._dict_to_ast(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(node, key, value)
                
        return node
    
    def _get_node_source(self, node: Union[Dict[str, Any], ASTNode]) -> str:
        """Get the source code for a node."""
        try:
            if isinstance(node, dict):
                if '_source' in node and node['_source']:
                    return node['_source']
                # Fallback: reconstruct from dict
                node = self._dict_to_ast(node)
                
            if HAS_ASTOR:
                return astor.to_source(node).strip()
            else:
                return ast.unparse(node).strip()  # Python 3.9+
        except Exception as e:
            self.logger.warning(f"Could not get source for node: {str(e)}")
            return f"# Source unavailable: {str(e)}"
    
    def _check_long_functions(self, 
                            node: Union[Dict[str, Any], ASTNode], 
                            suggestions: List[RefactoringSuggestion],
                            file_path: str) -> None:
        """Check for functions that are too long and should be split."""
        max_lines = 50  # Default threshold, could be configurable
        
        def process_node(node: Union[Dict[str, Any], ASTNode]) -> None:
            node_type = node['type'] if isinstance(node, dict) else node.__class__.__name__
            
            if node_type == 'FunctionDef':
                # Calculate function length in lines
                start_line = node.lineno if hasattr(node, 'lineno') else 1
                end_line = getattr(node, 'end_lineno', start_line)
                length = end_line - start_line + 1
                
                if length > max_lines:
                    # Get function source code
                    func_code = self._get_node_source(node)
                    func_name = getattr(node, 'name', 'anonymous')
                    
                    # Create suggestion to split the function
                    suggestion = RefactoringSuggestion(
                        refactoring_type=RefactoringType.SPLIT_FUNCTION,
                        description=f"Function '{func_name}' is too long ({length} lines). Consider splitting it into smaller functions.",
                        location=(start_line, end_line),
                        code_before=func_code,
                        code_after=None,  # We can't auto-generate the perfect split
                        confidence=min(0.9, 0.5 + (length - max_lines) / 100),  # Scale confidence with length
                        metadata={
                            'function_name': func_name,
                            'line_count': length,
                            'max_recommended_lines': max_lines,
                            'file': file_path
                        }
                    )
                    suggestions.append(suggestion)
            
            # Recursively process child nodes
            for key, value in node.items():
                if isinstance(value, dict):
                    process_node(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            process_node(item)
        
        process_node(ast_data)
    
    def _check_unused_variables(self, 
                              node: Union[Dict[str, Any], ASTNode], 
                              suggestions: List[RefactoringSuggestion],
                              file_path: str) -> None:
        """Check for unused variables in the code."""
        class VariableUsageTracker(ast.NodeVisitor):
            def __init__(self):
                self.assigned = set()
                self.used = set()
                
            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assigned.add(target.id)
                self.generic_visit(node)
                
            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, ast.Load):
                    self.used.add(node.id)
                self.generic_visit(node)
        
        try:
            # Convert to AST node if it's a dict
            if isinstance(node, dict):
                node = self._dict_to_ast(node)
                
            # Track variable usage
            tracker = VariableUsageTracker()
            tracker.visit(node)
            
            # Find unused variables
            unused_vars = tracker.assigned - tracker.used
            
            # Create suggestions for unused variables
            for var_name in unused_vars:
                if var_name.startswith('_'):
                    continue  # Skip private variables
                    
                # Find where this variable is assigned
                for node in ast.walk(node):
                    if (isinstance(node, ast.Assign) and 
                        any(isinstance(t, ast.Name) and t.id == var_name for t in ast.walk(node) 
                            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store))):
                        
                        suggestion = RefactoringSuggestion(
                            refactoring_type=RefactoringType.RENAME_VARIABLE,
                            description=f"Unused variable '{var_name}'. Consider removing or using it.",
                            location=(node.lineno, getattr(node, 'end_lineno', node.lineno)),
                            code_before=self._get_node_source(node),
                            code_after=None,  # Just remove the line
                            confidence=0.8,
                            metadata={
                                'variable_name': var_name,
                                'suggestion': 'Remove or use this variable.',
                                'file': file_path
                            }
                        )
                        suggestions.append(suggestion)
                        break
                        
        except Exception as e:
            self.logger.warning(f"Error checking for unused variables: {str(e)}")
    
    def _check_duplicate_code(self, 
                            node: Union[Dict[str, Any], ASTNode], 
                            suggestions: List[RefactoringSuggestion],
                            file_path: str) -> None:
        """Check for duplicate code blocks that could be refactored."""
        # This is a simplified implementation
        # A real implementation would use more sophisticated code clone detection
        
        try:
            # Convert to AST node if it's a dict
            if isinstance(node, dict):
                node = self._dict_to_ast(node)
            
            # Track function bodies to find duplicates
            function_bodies = {}
            
            class FunctionVisitor(ast.NodeVisitor):
                def __init__(self, refactoring_engine):
                    self.refactoring_engine = refactoring_engine
                    self.function_bodies = {}
                    
                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    # Get function body as a string for comparison
                    body_str = self.refactoring_engine._get_node_source(node)
                    
                    # Check for similar functions
                    for other_name, (other_body, other_loc) in self.function_bodies.items():
                        if body_str == other_body:
                            # Found a potential duplicate
                            suggestion = RefactoringSuggestion(
                                refactoring_type=RefactoringType.REMOVE_DUPLICATE_CODE,
                                description=f"Function '{node.name}' appears to be similar to '{other_name}'. Consider extracting common code.",
                                location=(node.lineno, getattr(node, 'end_lineno', node.lineno)),
                                code_before=self.refactoring_engine._get_node_source(node),
                                code_after=None,  # Would show extracted function in a real implementation
                                confidence=0.75,
                                metadata={
                                    'similar_function': other_name,
                                    'similar_location': other_loc,
                                    'file': file_path
                                }
                            )
                            suggestions.append(suggestion)
                            break
                    
                    # Store this function for future comparison
                    self.function_bodies[node.name] = (
                        body_str,
                        (node.lineno, getattr(node, 'end_lineno', node.lineno))
                    )
                    
                    self.generic_visit(node)
            
            # Run the visitor
            visitor = FunctionVisitor(self)
            visitor.visit(node)
            
        except Exception as e:
            self.logger.warning(f"Error checking for duplicate code: {str(e)}")
    
    def apply_refactoring(self, code: str, suggestion: RefactoringSuggestion) -> str:
        """Apply a refactoring suggestion to the code.
        
        Args:
            code: The original source code.
            suggestion: The refactoring suggestion to apply.
            
        Returns:
            A tuple of (refactored_code: str, success: bool, message: str)
        """
        try:
            # Parse the code to AST
            tree = ast.parse(code)
            
            # Apply the appropriate refactoring
            if suggestion.refactoring_type == RefactoringType.REMOVE_UNUSED_IMPORTS:
                new_tree = self._remove_unused_imports(tree)
            elif suggestion.refactoring_type == RefactoringType.RENAME_VARIABLE:
                new_tree = self._rename_variable(
                    tree, 
                    suggestion.metadata['old_name'],
                    suggestion.metadata['new_name']
                )
            else:
                # For other refactorings, use the provided code_after if available
                if not suggestion.code_after:
                    return code, False, "No automatic fix available for this refactoring"
                
                # Parse the refactored code
                new_tree = ast.parse(suggestion.code_after)
            
            # Convert back to source code
            if HAS_ASTOR:
                return astor.to_source(new_tree), True, "Refactoring applied successfully"
            else:
                return ast.unparse(new_tree), True, "Refactoring applied successfully"
                
        except Exception as e:
            self.logger.error(f"Error applying refactoring: {str(e)}", exc_info=True)
            return code, False, f"Error applying refactoring: {str(e)}"
    
    def _remove_unused_imports(self, tree: ast.AST) -> ast.AST:
        """Remove unused imports from the AST."""
        # Find all imports and track their usage
        import_nodes = []
        used_names = set()
        
        class ImportVisitor(ast.NodeVisitor):
            def __init__(self):
                self.imports = []
                self.used = set()
                
            def visit_Import(self, node: ast.Import) -> None:
                self.imports.append(node)
                self.generic_visit(node)
                
            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                self.imports.append(node)
                self.generic_visit(node)
                
            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, ast.Load):
                    self.used.add(node.id)
                self.generic_visit(node)
        
        # Find imports and used names
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        # Filter out unused imports
        new_body = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Skip if it's an unused import
                if not any(alias.name in visitor.used for alias in (node.names if hasattr(node, 'names') else [])):
                    continue
            new_body.append(node)
        
        # Create a new AST with unused imports removed
        new_tree = ast.Module(
            body=new_body,
            type_ignores=[]
        )
        
        return new_tree
    
    def _rename_variable(self, tree: ast.AST, old_name: str, new_name: str) -> ast.AST:
        """Rename a variable in the AST."""
        class RenameTransformer(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.Name:
                if node.id == old_name:
                    node.id = new_name
                return node
        
        return RenameTransformer().visit(tree)
