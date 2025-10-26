"""
Core functionality for the Code Explainer Agent.

This module contains the main agent implementation and configuration.
"""

from .agent import CodeExplainerAgent
from .config import AgentConfig, Language, CodeAnalysisConfig, DocumentationConfig, RefactoringConfig, EmbeddingConfig

__all__ = [
    'CodeExplainerAgent',
    'AgentConfig',
    'Language',
    'CodeAnalysisConfig',
    'DocumentationConfig',
    'RefactoringConfig',
    'EmbeddingConfig'
]
