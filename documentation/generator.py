"""
Documentation generator for the Code Explainer Agent.
"""
import os
import re
import ast
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, cast
import textwrap
import logging

from ..core.config import DocumentationConfig

# Set up logging
logger = logging.getLogger(__name__)

# Try to import astor, but make it optional
try:
    import astor
    HAS_ASTOR = True
except ImportError:
    HAS_ASTOR = False

class DocumentationGenerator:
    """
    Generates documentation for code elements based on AST analysis.
    """
    
    def __init__(self, config: DocumentationConfig):
        """Initialize the documentation generator.
        
        Args:
            config: Documentation configuration.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        if not isinstance(config, DocumentationConfig):
            raise ValueError("config must be an instance of DocumentationConfig")
            
        self.config = config
        self._templates = self._load_templates()
        
        # Validate templates
        required_templates = {'module', 'class', 'function'}
        for style, style_templates in self._templates.items():
            missing = required_templates - set(style_templates.keys())
            if missing:
                logger.warning(f"Style '{style}' is missing templates: {missing}")
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load documentation templates based on style.
        
        Returns:
            Dictionary containing templates for different docstring styles.
            Each style has templates for 'module', 'class', and 'function'.
            
        Raises:
            ValueError: If the configured style is not supported.
        """
        # Default templates for different docstring styles
        templates = {
            'google': {
                'module': """# {module_name}

{docstring}

## Classes

{classes}

## Functions

{functions}
""",
                'class': '''class {name}{bases}:
    """{docstring}"""
    
{methods}
''',
                'function': '''def {name}({params}) -> {return_type}:
    """{docstring}
    
    Args:
{args_doc}
    
    Returns:
        {return_type}: {returns_doc}
    """
    ...
'''
            },
            'numpy': {
                'module': """# {module_name}

{docstring}

## Classes

{classes}

## Functions

{functions}
""",
                'class': '''class {name}{bases}:
    """{docstring}"""
    
{methods}
''',
                'function': '''def {name}({params}) -> {return_type}:
    """{docstring}
    
    Parameters
    ----------
{args_doc}
    
    Returns
    -------
    {return_type}
        {returns_doc}
    """
    ...
'''
            },
            'rst': {
                'module': """{module_name}
{underline}

{docstring}

Classes
-------

{classes}

Functions
---------

{functions}
""",
                'class': """{name}{bases}
{underline}

{docstring}

.. py:class:: {name}{bases}

{docstring}
""",
                'function': """{name}
{underline}

.. py:function:: {name}({params}) -> {return_type}
   :noindex:

   {docstring}
   
   {param_docs}
   
   :return: {returns_doc}
   :rtype: {return_type}
"""
            }
        }
        
        if self.config.style not in templates:
            logger.warning(f"Unknown documentation style: {self.config.style}. Using 'google' style.")
            return templates['google']
            
        return templates[self.config.style]
    
    def _get_node_source(self, node: Union[Dict[str, Any], ast.AST]) -> str:
        """Get the source code for a node."""
        try:
            if HAS_ASTOR and not isinstance(node, dict):
                return astor.to_source(node).strip()
            elif hasattr(ast, 'unparse') and not isinstance(node, dict):
                return ast.unparse(node).strip()
            elif isinstance(node, dict) and '_source' in node:
                return node['_source']
            else:
                return "# Source code not available"
        except Exception as e:
            logger.warning(f"Could not get source for node: {str(e)}")
            return "# Error getting source code"
    
    def _get_annotation_str(self, node: Union[Dict[str, Any], ast.AST]) -> str:
        """Convert an annotation node to a string."""
        if node is None:
            return "Any"
            
        if isinstance(node, dict):
            node_type = node.get('type')
            
            if node_type == 'Name':
                return node.get('id', 'Any')
            elif node_type == 'Subscript':
                value = self._get_annotation_str(node.get('value'))
                slice_val = self._get_annotation_str(node.get('slice'))
                return f"{value}[{slice_val}]"
            elif node_type == 'Index':
                return self._get_annotation_str(node.get('value'))
            elif node_type == 'Constant':
                return str(node.get('value', 'Any'))
            elif node_type == 'Attribute':
                value = self._get_annotation_str(node.get('value'))
                return f"{value}.{node.get('attr', 'Any')}"
            else:
                return f"<{node_type}>"
                
        # Handle AST nodes directly
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_str(node.value)
            slice_val = self._get_annotation_str(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Index):
            return self._get_annotation_str(node.value)
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Attribute):
            value = self._get_annotation_str(node.value)
            return f"{value}.{node.attr}"
            
        return "Any"
    
    def generate_documentation(self, ast_data: Union[Dict[str, Any], ast.AST], file_path: str) -> str:
        """Generate documentation for a parsed file.
        
        Args:
            ast_data: The parsed AST data (can be dict or AST node).
            file_path: Path to the source file.
            
        Returns:
            Generated documentation as a string.
        """
        try:
            if ast_data is None:
                return "# Error: No AST data provided"
                
            # Convert dict to AST node if needed
            if isinstance(ast_data, dict) and 'type' not in ast_data:
                return "# Error: Invalid AST data"
            
            # Extract module-level information
            module_name = Path(file_path).stem
            docstring = self._extract_docstring(ast_data)
            
            # Extract classes and functions
            classes = []
            functions = []
            
            # Handle both dict and AST node representations
            body = []
            if isinstance(ast_data, dict):
                body = ast_data.get('body', [])
            elif hasattr(ast_data, 'body'):
                body = ast_data.body
            
            for node in body:
                node_type = node.get('type') if isinstance(node, dict) else node.__class__.__name__
                
                if node_type == 'ClassDef' or (isinstance(node, ast.ClassDef)):
                    classes.append(self._document_class(node))
                elif node_type == 'FunctionDef' or (isinstance(node, ast.FunctionDef)):
                    functions.append(self._document_function(node))
            
            # Generate module documentation
            module_doc = self._templates['module'].format(
                summary=f"# {module_name}\n\nModule documentation",
                docstring=f"""{docstring}""" if docstring else "",
                classes="\n\n".join(classes) if classes else "",
                functions="\n\n".join(functions) if functions else ""
            )
            
            return module_doc
            
        except Exception as e:
            logger.error(f"Error generating documentation: {str(e)}", exc_info=True)
            return f"# Error generating documentation: {str(e)}"
    
    def _document_class(self, node: Union[Dict[str, Any], ast.ClassDef]) -> str:
        """Generate documentation for a class."""
        try:
            # Handle both dict and AST node representations
            if isinstance(node, dict):
                class_name = node.get('name', 'UnknownClass')
                bases = [self._get_annotation_str(b) for b in node.get('bases', [])]
                body = node.get('body', [])
                docstring = self._extract_docstring(node)
            else:
                class_name = getattr(node, 'name', 'UnknownClass')
                bases = [self._get_annotation_str(b) for b in getattr(node, 'bases', [])]
                body = getattr(node, 'body', [])
                docstring = self._extract_docstring(node)
            
            # Process methods
            methods = []
            for item in body:
                item_type = item.get('type') if isinstance(item, dict) else item.__class__.__name__
                if item_type == 'FunctionDef' or isinstance(item, ast.FunctionDef):
                    methods.append(self._document_function(item, is_method=True))
            
            # Format class signature
            bases_str = f"({', '.join(bases)})" if bases else ""
            class_signature = f"class {class_name}{bases_str}:"
            
            # Format docstring
            docstring = textwrap.indent(f'"""{docstring or "Class docstring."}"""', '    ')
            
            # Combine everything
            result = [
                class_signature,
                docstring,
                ""
            ]
            
            if methods:
                result.append("## Methods\n")
                result.extend(methods)
            
            return "\n".join(result)
            
        except Exception as e:
            logger.error(f"Error documenting class: {str(e)}", exc_info=True)
            return f"# Error documenting class: {str(e)}"
    
    def _document_function(self, node: Union[Dict[str, Any], ast.FunctionDef], is_method: bool = False) -> str:
        """Generate documentation for a function or method."""
        try:
            # Handle both dict and AST node representations
            if isinstance(node, dict):
                func_name = node.get('name', 'unknown_function')
                args = node.get('args', {})
                args_list = args.get('args', [])
                return_annotation = node.get('returns')
                docstring = self._extract_docstring(node)
                decorator_list = node.get('decorator_list', [])
            else:
                func_name = getattr(node, 'name', 'unknown_function')
                args = getattr(node, 'args', ast.arguments(args=[]))
                args_list = getattr(args, 'args', [])
                return_annotation = getattr(node, 'returns', None)
                docstring = self._extract_docstring(node)
                decorator_list = getattr(node, 'decorator_list', [])
            
            # Process parameters
            params = []
            args_doc = []
            
            # Handle regular arguments
            for arg in args_list:
                if isinstance(arg, dict):
                    arg_name = arg.get('arg', 'unknown')
                    arg_type = self._get_annotation_str(arg.get('annotation'))
                else:
                    arg_name = getattr(arg, 'arg', 'unknown')
                    arg_type = self._get_annotation_str(getattr(arg, 'annotation', None))
                
                # Skip 'self' for methods
                if is_method and arg_name == 'self':
                    continue
                
                # Add to parameters list
                param_str = arg_name
                if arg_type:
                    param_str += f": {arg_type}"
                
                # Check for default value
                default = None
                if isinstance(arg, dict):
                    if 'default' in arg and arg['default'] is not None:
                        default = self._get_node_source(arg['default'])
                elif hasattr(arg, 'default') and arg.default is not None:
                    default = self._get_node_source(arg.default)
                
                if default is not None:
                    param_str += f" = {default}"
                
                params.append(param_str)
                
                # Add to args documentation based on style
                if self.config.style == 'google':
                    arg_doc = f"        {arg_name} ({arg_type or 'Any'}): Description of {arg_name}."
                    if default is not None:
                        arg_doc += f" Defaults to {default}."
                    args_doc.append(arg_doc)
                    
                elif self.config.style == 'numpy':
                    args_doc.append(f"    {arg_name} : {arg_type or 'Any'}")
                    args_doc.append(f"        Description of {arg_name}.")
                    if default is not None:
                        args_doc[-1] += f" Defaults to {default}."
                        
                else:  # rst
                    args_doc.append(f"    :param {arg_name}: Description of {arg_name}.")
                    if arg_type:
                        args_doc.append(f"    :type {arg_name}: {arg_type}")
            
            # Handle return type
            return_type = self._get_annotation_str(return_annotation)
            returns_doc = return_type or "Any"
            
            # Generate function signature
            decorators = ""
            if decorator_list:
                decorator_strs = []
                for dec in decorator_list:
                    if isinstance(dec, dict):
                        if dec.get('type') == 'Name':
                            decorator_strs.append(f"@{dec.get('id', '')}")
                        elif dec.get('type') == 'Call' and dec.get('func', {}).get('id'):
                            decorator_strs.append(f"@{dec['func']['id']}()")
                    elif hasattr(dec, 'id'):
                        decorator_strs.append(f"@{dec.id}")
                    elif hasattr(dec, 'func') and hasattr(dec.func, 'id'):
                        decorator_strs.append(f"@{dec.func.id}()")
                
                if decorator_strs:
                    decorators = "\n".join(decorator_strs) + "\n"
            
            # Format function signature
            func_signature = f"def {func_name}({', '.join(params)}) -> {returns_doc}:"
            
            # Format docstring
            if not docstring:
                docstring = f"{'Method' if is_method else 'Function'} {func_name}."
            
            # Build the documentation
            if self.config.style == 'google':
                doc_parts = [
                    f"{decorators}{func_signature}",
                    f'    """{docstring}',
                    ""
                ]
                
                if args_doc:
                    doc_parts.extend([
                        "    Args:",
                        *[f"{line}" for line in args_doc],
                        ""
                    ])
                
                doc_parts.extend([
                    f"    Returns:",
                    f"        {returns_doc}: Description of return value.",
                    '    """',
                    "    ..."
                ])
                
                return "\n".join(doc_parts)
                
            elif self.config.style == 'numpy':
                doc_parts = [
                    f"{decorators}{func_signature}",
                    f'    """{docstring}',
                    ""
                ]
                
                if args_doc:
                    doc_parts.append("    Parameters")
                    doc_parts.append("    ----------")
                    doc_parts.extend(args_doc)
                    doc_parts.append("")
                
                doc_parts.extend([
                    "    Returns",
                    "    -------",
                    f"    {returns_doc}",
                    "        Description of return value.",
                    '    """',
                    "    ..."
                ])
                
                return "\n".join(doc_parts)
                
            else:  # rst
                doc_parts = [
                    f"{decorators}{func_signature}",
                    f'    """{docstring}',
                    ""
                ]
                
                if args_doc:
                    doc_parts.extend(args_doc)
                    doc_parts.append("")
                
                doc_parts.extend([
                    f"    :return: Description of return value.",
                    f"    :rtype: {returns_doc}",
                    '    """',
                    "    ..."
                ])
                
                return "\n".join(doc_parts)
                
        except Exception as e:
            logger.error(f"Error documenting function: {str(e)}", exc_info=True)
            return f"# Error documenting function: {str(e)}"
    
    def _extract_docstring(self, node: Union[Dict[str, Any], ast.AST]) -> str:
        """Extract docstring from a node if it exists."""
        try:
            if isinstance(node, dict):
                if not node.get('body'):
                    return ""
                
                first_node = node['body'][0]
                if (isinstance(first_node, dict) and 
                    first_node.get('type') == 'Expr' and 
                    isinstance(first_node.get('value'), dict) and 
                    first_node['value'].get('type') in ('Str', 'Constant')):
                    
                    # Handle Python 3.8+ Constant node or older Str node
                    value = first_node['value']
                    if value['type'] == 'Str':
                        return value.get('s', '')
                    elif value['type'] == 'Constant':
                        if isinstance(value.get('value'), str):
                            return value['value']
            
            # Handle AST nodes directly
            elif isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                if not node.body:
                    return ""
                    
                first_node = node.body[0]
                if (isinstance(first_node, ast.Expr) and 
                    isinstance(first_node.value, (ast.Str, ast.Constant))):
                    
                    if isinstance(first_node.value, ast.Str):
                        return first_node.value.s
                    elif isinstance(first_node.value.value, str):
                        return first_node.value.value
                        
        except Exception as e:
            logger.warning(f"Error extracting docstring: {str(e)}")
            
        return ""
    

