"""
Configuration settings - FIXED VERSION
"""
from enum import Enum
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"

class CodeAnalysisConfig(BaseModel):
    """Configuration for code analysis."""
    enable_complexity_analysis: bool = True
    enable_code_smell_detection: bool = True
    max_cyclomatic_complexity: int = 10
    max_cognitive_complexity: int = 15
    max_function_length: int = 50
    max_parameters: int = 5

class DocumentationConfig(BaseModel):
    """Configuration for documentation generation."""
    style: str = "google"
    generate_examples: bool = True
    include_type_hints: bool = True
    generate_uml: bool = True
    output_format: str = "markdown"

class RefactoringConfig(BaseModel):
    """Configuration for code refactoring."""
    auto_apply: bool = False
    suggest_renames: bool = True
    split_long_functions: bool = True
    remove_unused_variables: bool = True
    remove_duplicate_code: bool = True

class EmbeddingConfig(BaseModel):
    """Configuration for code embeddings."""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    batch_size: int = 32

class AgentConfig(BaseModel):
    """Main configuration for the Code Explainer Agent."""
    project_root: Path = Path(".")
    language: Language = Language.PYTHON
    analysis: CodeAnalysisConfig = Field(default_factory=CodeAnalysisConfig)
    documentation: DocumentationConfig = Field(default_factory=DocumentationConfig)
    refactoring: RefactoringConfig = Field(default_factory=RefactoringConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    output_dir: Path = Path("output")
    cache_dir: Path = Path(".cache")
    
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_api_key: Optional[str] = None
    temperature: float = 0.2
    
    enable_memory: bool = False
    enable_self_reflection: bool = False
    max_iterations: int = 3
    
    def model_post_init(self, __context):
        """Initialize directories after model creation."""
        try:
            self.project_root = Path(self.project_root).resolve()
            self.output_dir = self.project_root / self.output_dir
            self.cache_dir = self.project_root / self.cache_dir
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.warning(f"Could not initialize directories: {str(e)}")
        
        # Handle API keys
        if self.llm_provider == "gemini":
            if self.llm_api_key is None:
                self.llm_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not self.llm_api_key:
                    logging.warning("No Gemini API key found. Set GEMINI_API_KEY environment variable.")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Path: str,
        }  # FIXED: Added closing brace

import logging
