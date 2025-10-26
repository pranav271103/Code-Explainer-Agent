"""
Base parser interface for different programming languages.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class CodeParser(ABC):
    """Abstract base class for code parsers."""
    
    @abstractmethod
    def parse(self, code: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Parse the given code and return an AST.
        
        Args:
            code: Source code to parse.
            file_path: Optional path to the source file (for better error messages).
            
        Returns:
            A dictionary representing the parsed AST.
        """
        pass
    
    @abstractmethod
    def extract_functions(self, ast: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract function definitions from the AST.
        
        Args:
            ast: The parsed AST.
            
        Returns:
            Dictionary mapping function names to their metadata.
        """
        pass
    
    @abstractmethod
    def extract_classes(self, ast: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract class definitions from the AST.
        
        Args:
            ast: The parsed AST.
            
        Returns:
            Dictionary mapping class names to their metadata.
        """
        pass
    
    @abstractmethod
    def get_node_source(self, node: Dict[str, Any], source: str) -> str:
        """Get the source code for a specific AST node.
        
        Args:
            node: The AST node.
            source: The full source code.
            
        Returns:
            The source code corresponding to the node.
        """
        pass
