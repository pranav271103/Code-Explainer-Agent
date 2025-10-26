"""
Embedding manager for code embeddings and semantic search with FAISS integration.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import hashlib
import json
import os
import faiss
import pickle
from dataclasses import dataclass, asdict, field
from tqdm import tqdm

from ..core.config import EmbeddingConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class CodeChunk:
    """Represents a chunk of code with its metadata and embedding."""
    code: str
    identifier: str
    start_line: int
    end_line: int
    file_path: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            'code': self.code,
            'identifier': self.identifier,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'file_path': self.file_path,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeChunk':
        """Create a CodeChunk from a dictionary."""
        return cls(
            code=data['code'],
            identifier=data['identifier'],
            start_line=data['start_line'],
            end_line=data['end_line'],
            file_path=data['file_path'],
            embedding=np.array(data['embedding']) if data['embedding'] is not None else None,
            metadata=data.get('metadata', {})
        )


class EmbeddingManager:
    """
    Manages code embeddings with FAISS for efficient similarity search.
    
    Features:
    - FAISS for fast similarity search
    - Chunking of large code files
    - Persistent storage of embeddings
    - Batch processing
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedding manager.
        
        Args:
            config: Embedding configuration.
        """
        self.config = config
        self._model = self._load_embedding_model()
        self._cache: Dict[str, np.ndarray] = {}
        self._chunks: Dict[str, CodeChunk] = {}
        self._index = None
        self._is_index_trained = False
        self._init_faiss_index()
        
    def _init_faiss_index(self) -> None:
        """Initialize the FAISS index."""
        # We'll use a flat index for simplicity (exact search)
        # For larger datasets, consider using IndexIVFFlat or IndexHNSWFlat
        self._index = faiss.IndexFlatL2(self.config.embedding_dimension)
        self._is_index_trained = True  # Flat index doesn't need training
        
    def _load_embedding_model(self):
        """Load the embedding model based on configuration."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.config.model_name}")
            model = SentenceTransformer(self.config.model_name)
            
            # Set embedding dimension if not already set
            if not hasattr(self.config, 'embedding_dimension'):
                # Get embedding dimension from the model
                test_embedding = model.encode(['test'])[0]
                self.config.embedding_dimension = len(test_embedding)
                
            return model
            
        except ImportError as e:
            logger.error("sentence-transformers is required for embeddings")
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e
    
    def chunk_code(
        self,
        code: str,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[CodeChunk]:
        """Split code into chunks with optional overlap.
        
        Args:
            code: Source code to chunk.
            file_path: Path to the source file.
            chunk_size: Maximum number of lines per chunk.
            chunk_overlap: Number of lines to overlap between chunks.
            
        Returns:
            List of CodeChunk objects.
        """
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.config.chunk_overlap
            
        lines = code.splitlines()
        chunks = []
        
        for i in range(0, len(lines), chunk_size - chunk_overlap):
            chunk_lines = lines[i:i + chunk_size]
            chunk_code = '\n'.join(chunk_lines)
            
            chunk = CodeChunk(
                code=chunk_code,
                identifier=f"{file_path}:{i+1}-{min(i + chunk_size, len(lines))}",
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines)),
                file_path=file_path,
                metadata={
                    'chunk_index': len(chunks),
                    'total_chunks': (len(lines) + chunk_size - 1) // chunk_size
                }
            )
            chunks.append(chunk)
            
        return chunks
        
    def embed_code(
        self,
        code: str,
        identifier: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Generate an embedding for the given code snippet.
        
        Args:
            code: Source code to embed.
            identifier: Unique identifier for the code (e.g., file path).
            metadata: Optional metadata to store with the embedding.
            
        Returns:
            Embedding vector as a numpy array.
        """
        # Check cache first
        cache_key = self._get_cache_key(code, identifier)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Generate embedding
            embedding = self._model.encode(
                code,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Cache the result
            self._cache[cache_key] = embedding
            
            # Create a chunk if metadata is provided
            if metadata is not None:
                chunk = CodeChunk(
                    code=code,
                    identifier=identifier,
                    start_line=metadata.get('start_line', 0),
                    end_line=metadata.get('end_line', 0),
                    file_path=metadata.get('file_path', ''),
                    embedding=embedding,
                    metadata=metadata
                )
                self._chunks[cache_key] = chunk
                
                # Add to FAISS index if it's initialized
                if self._index is not None and self._is_index_trained:
                    self._index.add(np.array([embedding]))
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def batch_embed_code(self, code_chunks: List[Tuple[str, str]]) -> List[np.ndarray]:
        """Generate embeddings for multiple code chunks.
        
        Args:
            code_chunks: List of (code, identifier) tuples.
            
        Returns:
            List of embedding vectors.
        """
        # Separate cached and uncached items
        uncached_chunks = []
        results = [None] * len(code_chunks)
        
        for i, (code, identifier) in enumerate(code_chunks):
            cache_key = self._get_cache_key(code, identifier)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_chunks.append((i, code, identifier))
        
        # Process uncached chunks in batches
        if uncached_chunks:
            # Prepare batch
            batch_indices, batch_codes, batch_identifiers = zip(*uncached_chunks)
            
            # Generate embeddings
            batch_embeddings = self._model.encode(
                batch_codes,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Update cache and results
            for idx, embedding in zip(batch_indices, batch_embeddings):
                code, identifier = code_chunks[idx]
                cache_key = self._get_cache_key(code, identifier)
                self._cache[cache_key] = embedding
                results[idx] = embedding
        
        return results
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Similarity score between 0 and 1.
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Calculate cosine similarity
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def find_similar_code(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.7,
        filter_func: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Find code chunks similar to the query using FAISS for efficient search.
        
        Args:
            query: The query string or code snippet.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score to include in results.
            filter_func: Optional function to filter chunks before searching.
                        Should take a CodeChunk and return a boolean.
                        
        Returns:
            List of dictionaries containing matching chunks and their similarity scores.
        """
        if not self._chunks:
            logger.warning("No embeddings available. Please add some code chunks first.")
            return []
            
        # Generate query embedding
        query_embedding = self.embed_code(query, "__query__")
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Filter chunks if needed
        chunks_to_search = list(self._chunks.values())
        if filter_func is not None:
            chunks_to_search = [chunk for chunk in chunks_to_search if filter_func(chunk)]
            
        if not chunks_to_search:
            return []
            
        # Get embeddings for all chunks
        embeddings = np.array([chunk.embedding for chunk in chunks_to_search], dtype=np.float32)
        
        # Use FAISS for efficient search if available
        if self._index is not None and self._is_index_trained:
            # Get distances and indices of nearest neighbors
            distances, indices = self._index.search(query_embedding, min(top_k, len(chunks_to_search)))
            
            # Convert to results format
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(chunks_to_search):
                    continue
                    
                chunk = chunks_to_search[idx]
                score = 1.0 / (1.0 + distances[0][i])  # Convert L2 distance to similarity
                
                if score >= threshold:
                    result = {
                        'chunk': chunk,
                        'score': float(score),
                        'code': chunk.code,
                        'file_path': chunk.file_path,
                        'start_line': chunk.start_line,
                        'end_line': chunk.end_line,
                        'metadata': chunk.metadata
                    }
                    results.append(result)
                    
            # Sort by score (descending)
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        else:
            # Fallback to linear search if FAISS is not available
            logger.warning("FAISS index not available, falling back to linear search")
            return self._linear_similarity_search(query_embedding[0], chunks_to_search, top_k, threshold)
    
    def _linear_similarity_search(
        self,
        query_embedding: np.ndarray,
        chunks: List[CodeChunk],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Fallback linear similarity search."""
        similarities = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                score = self.similarity(query_embedding, chunk.embedding)
                if score >= threshold:
                    similarities.append((chunk, score))
        
        # Sort by score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to result format
        return [
            {
                'chunk': chunk,
                'score': score,
                'code': chunk.code,
                'file_path': chunk.file_path,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'metadata': chunk.metadata
            }
            for chunk, score in similarities[:top_k]
        ]
    
    def save_embeddings(self, file_path: str) -> None:
        """Save embeddings and FAISS index to disk.
        
        Args:
            file_path: Base path for saving files (will create multiple files).
        """
        # Create parent directory if it doesn't exist
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self._index is not None and self._is_index_trained:
            faiss.write_index(self._index, f"{file_path}.faiss")
        
        # Save chunks and metadata
        chunks_data = {
            'config': {
                'model_name': self.config.model_name,
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'embedding_dimension': self.config.embedding_dimension
            },
            'chunks': [chunk.to_dict() for chunk in self._chunks.values()]
        }
        
        with open(f"{file_path}.json", 'w') as f:
            json.dump(chunks_data, f, indent=2, default=str)
            
        logger.info(f"Saved {len(self._chunks)} embeddings to {file_path}.*")
    
    def load_embeddings(self, file_path: str) -> None:
        """Load embeddings and FAISS index from disk.
        
        Args:
            file_path: Base path for loading files.
        """
        # Check if files exist
        if not os.path.exists(f"{file_path}.json"):
            logger.warning(f"Embeddings file not found: {file_path}.json")
            return
            
        try:
            # Load chunks and metadata
            with open(f"{file_path}.json", 'r') as f:
                data = json.load(f)
                
            # Update config if needed
            if 'config' in data:
                for key, value in data['config'].items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            # Load chunks
            self._chunks = {}
            for chunk_data in data.get('chunks', []):
                chunk = CodeChunk.from_dict(chunk_data)
                cache_key = self._get_cache_key(chunk.code, chunk.identifier)
                self._chunks[cache_key] = chunk
                self._cache[cache_key] = chunk.embedding
            
            # Load FAISS index if it exists
            if os.path.exists(f"{file_path}.faiss"):
                self._index = faiss.read_index(f"{file_path}.faiss")
                self._is_index_trained = True
                
            logger.info(f"Loaded {len(self._chunks)} embeddings from {file_path}.*")
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise
    
    def _get_cache_key(self, code: str, identifier: str) -> str:
        """Generate a cache key for the given code and identifier."""
        # Use a hash of the code and identifier for the cache key
        key = f"{identifier}:{code}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear the embedding cache and FAISS index."""
        self._cache.clear()
        self._chunks.clear()
        self._init_faiss_index()
        
    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Get a code chunk by its ID."""
        return self._chunks.get(chunk_id)
    
    def get_chunk_count(self) -> int:
        """Get the number of chunks in the index."""
        return len(self._chunks)
    
    def rebuild_index(self) -> None:
        """Rebuild the FAISS index from the current chunks."""
        if not self._chunks:
            logger.warning("No chunks available to index")
            return
            
        # Get all embeddings
        embeddings = []
        valid_chunks = []
        
        for chunk in self._chunks.values():
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)
        
        if not embeddings:
            logger.warning("No valid embeddings found to index")
            return
            
        # Convert to numpy array
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Create and train new index
        self._index = faiss.IndexFlatL2(embeddings.shape[1])
        self._index.add(embeddings)
        self._is_index_trained = True
        
        logger.info(f"Rebuilt FAISS index with {len(valid_chunks)} embeddings")
