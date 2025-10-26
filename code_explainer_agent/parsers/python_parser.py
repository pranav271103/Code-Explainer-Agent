"""
Python code parser implementation using the ast module.
"""
import ast
import inspect
from typing import Dict, Any, Optional, List, Tuple
import textwrap

from .base import CodeParser


class PythonParser(CodeParser):
    """Python code parser implementation."""
    
    def parse(self, code: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Parse Python code and return an AST."""
        try:
            tree = ast.parse(code, filename=file_path or '<string>')
            return self._ast_to_dict(tree)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path or 'input'}: {str(e)}")
    
    def extract_functions(self, ast_dict: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract function definitions from the AST."""
        functions = {}
        
        def process_node(node: Dict[str, Any]):
            if node['type'] == 'FunctionDef':
                # Get function signature
                args = []
                for arg in node['args']['args']:
                    arg_info = {'name': arg['arg']}
                    if 'annotation' in arg and arg['annotation']:
                        arg_info['type'] = self._get_annotation(arg['annotation'])
                    args.append(arg_info)
                
                # Get return type annotation if present
                return_type = None
                if 'returns' in node and node['returns']:
                    return_type = self._get_annotation(node['returns'])
                
                # Calculate complexity metrics
                complexity = self._calculate_complexity(node)
                
                # Get docstring
                docstring = None
                if node['body'] and node['body'][0]['type'] == 'Expr' and \
                   isinstance(node['body'][0]['value'], ast.Str):
                    docstring = node['body'][0]['value'].s
                
                functions[node['name']] = {
                    'name': node['name'],
                    'args': args,
                    'returns': return_type,
                    'lineno': node['lineno'],
                    'end_lineno': getattr(node, 'end_lineno', None),
                    'docstring': docstring,
                    'complexity': complexity,
                    'source': self._get_node_source(node, node['_source'])
                }
            
            # Recursively process child nodes
            for key, value in node.items():
                if isinstance(value, dict):
                    process_node(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            process_node(item)
        
        process_node(ast_dict)
        return functions
    
    def extract_classes(self, ast_dict: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract class definitions from the AST."""
        classes = {}
        
        def process_node(node: Dict[str, Any]):
            if node['type'] == 'ClassDef':
                # Get class docstring
                docstring = None
                if node['body'] and node['body'][0]['type'] == 'Expr' and \
                   isinstance(node['body'][0]['value'], ast.Str):
                    docstring = node['body'][0]['value'].s
                
                # Get methods
                methods = {}
                for item in node['body']:
                    if item['type'] == 'FunctionDef':
                        method_name = item['name']
                        methods[method_name] = {
                            'name': method_name,
                            'lineno': item['lineno'],
                            'end_lineno': getattr(item, 'end_lineno', None),
                            'is_async': item.get('is_async', False),
                            'decorators': [d['id'] for d in item.get('decorator_list', []) 
                                         if d['type'] == 'Name']
                        }
                
                bases = []
                for base in node.get('bases', []):
                    if base['type'] == 'Name':
                        bases.append(base['id'])
                
                classes[node['name']] = {
                    'name': node['name'],
                    'bases': bases,
                    'docstring': docstring,
                    'methods': methods,
                    'lineno': node['lineno'],
                    'end_lineno': getattr(node, 'end_lineno', None),
                    'source': self._get_node_source(node, node['_source'])
                }
            
            # Recursively process child nodes
            for key, value in node.items():
                if isinstance(value, dict):
                    process_node(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            process_node(item)
        
        process_node(ast_dict)
        return classes
    
    def get_node_source(self, node: Dict[str, Any], source: str) -> str:
        """Get the source code for a specific AST node."""
        return self._get_node_source(node, source)
    
    def _ast_to_dict(self, node, source: Optional[str] = None) -> Dict[str, Any]:
        """Convert an AST node to a dictionary."""
        if not isinstance(node, ast.AST):
            return node
        
        result = {
            'type': node.__class__.__name__,
            '_source': source
        }
        
        # Add line number information if available
        for attr in ['lineno', 'col_offset', 'end_lineno', 'end_col_offset']:
            if hasattr(node, attr):
                result[attr] = getattr(node, attr)
        
        # Add node attributes
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                result[field] = [self._ast_to_dict(item, source) for item in value]
            elif isinstance(value, ast.AST):
                result[field] = self._ast_to_dict(value, source)
            else:
                result[field] = value
        
        return result
    
    def _get_annotation(self, node: Dict[str, Any]) -> str:
        """Convert an AST annotation to a string."""
        if not node:
            return "Any"
        
        if node['type'] == 'Name':
            return node['id']
        elif node['type'] == 'Subscript':
            value = self._get_annotation(node['value'])
            slice_value = self._get_annotation(node['slice'])
            return f"{value}[{slice_value}]"
        elif node['type'] == 'Index':
            return self._get_annotation(node['value'])
        elif node['type'] == 'Tuple':
            elts = [self._get_annotation(elt) for elt in node['elts']]
            return f"({', '.join(elts)})"
        elif node['type'] == 'Constant':
            return repr(node['value'])
        elif node['type'] == 'Attribute':
            value = self._get_annotation(node['value'])
            return f"{value}.{node['attr']}"
        else:
            return f"<{node['type']}>"
    
    def _calculate_complexity(self, node: Dict[str, Any]) -> Dict[str, int]:
        """Calculate complexity metrics for a function node."""
        # This is a simplified implementation
        cyclomatic = 1  # Start with 1 for the function itself
        cognitive = 0
        
        def count_complexity(n: Dict[str, Any]):
            nonlocal cyclomatic, cognitive
            
            # Cyclomatic complexity components
            if n['type'] in ['If', 'For', 'While', 'And', 'Or', 'IfExp']:
                cyclomatic += 1
            elif n['type'] == 'BoolOp' and len(n['values']) > 1:
                cyclomatic += len(n['values']) - 1
            
            # Cognitive complexity components (simplified)
            if n['type'] in ['If', 'For', 'While', 'Try', 'With', 'ExceptHandler']:
                cognitive += 1
            
            # Recursively process child nodes
            for key, value in n.items():
                if isinstance(value, dict):
                    count_complexity(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            count_complexity(item)
        
        count_complexity(node)
        
        return {
            'cyclomatic': cyclomatic,
            'cognitive': cognitive,
            'lines': (node.get('end_lineno', node['lineno']) - node['lineno'] + 1) 
                     if 'lineno' in node else 0
        }
    
    def _get_node_source(self, node: Dict[str, Any], source: str) -> str:
        """Get the source code for a node."""
        if not source or 'lineno' not in node:
            return ""
        
        start_line = node['lineno'] - 1  # 0-based index
        end_line = node.get('end_lineno', start_line + 1) - 1
        
        lines = source.splitlines()
        if 0 <= start_line < len(lines) and 0 <= end_line < len(lines):
            return "\n".join(lines[start_line:end_line+1])
        return ""
