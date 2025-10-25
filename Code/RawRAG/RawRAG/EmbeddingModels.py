import logging
from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    """Abstract base class for all embedding models."""
    
    @abstractmethod
    def create_embedding(self, text, mode="passage"):
        """
        Create embedding for given text.
        
        Args:
            text: Input text to embed
            mode: Either "passage" or "query" for different encoding tasks
            
        Returns:
            Embedding vector as numpy array
        """
        pass


class SBertEmbeddingModel(BaseEmbeddingModel):
    """Sentence-BERT embedding model using sentence-transformers library."""
    
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        """
        Initialize SentenceTransformer model.
        
        Args:
            model_name: HuggingFace model name for sentence transformer
        """
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text, mode="passage"):
        """
        Create embedding using SentenceTransformer.
        
        Args:
            text: Input text to embed
            mode: Not used in SentenceTransformer (kept for interface compatibility)
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text)
    
    
class JinaEmbeddingModel(BaseEmbeddingModel):
    """Jina AI embedding model optimized for retrieval tasks."""
    
    def __init__(self, model_name="jinaai/jina-embeddings-v3"):
        """
        Initialize Jina embedding model (requires GPU).
        
        Args:
            model_name: Jina model name from HuggingFace
            
        Raises:
            RuntimeError: If CUDA is not available
        """
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            raise RuntimeError("JinaEmbeddingModel requires GPU to run. Please use SBertEmbeddingModel for CPU-only environments.")
            
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)

    def create_embedding(self, text, mode="passage"):
        """
        Create embedding using Jina model with task-specific encoding.
        
        Args:
            text: Input text to embed
            mode: Either "query" or "passage" for different retrieval tasks
            
        Returns:
            Embedding vector as numpy array
        """
        assert mode in ["query", "passage"]
        task = f"retrieval.{mode}"
        return self.model.encode(text, task=task)
    
    

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI embedding model using their API service."""
    
    def __init__(self, model="text-embedding-ada-002"):
        """
        Initialize OpenAI embedding model.
        
        Args:
            model: OpenAI embedding model name
        """
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text, mode=None):
        """
        Create embedding using OpenAI API with retry logic.
        
        Args:
            text: Input text to embed
            mode: Not used for OpenAI (kept for interface compatibility)
            
        Returns:
            Embedding vector from OpenAI API
        """
        # Clean text for API call
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )