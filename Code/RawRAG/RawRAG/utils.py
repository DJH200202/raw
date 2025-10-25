import logging
import re
from typing import List

from chonkie import TokenChunker, SentenceChunker, SemanticChunker
import numpy as np
import tiktoken
from scipy import spatial


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def split_text_by_token(text: str, max_tokens: int, overlap: int = 0):
    """
    Split text into chunks based on token count using GPT-2 tokenizer.
    
    Args:
        text: Input text to split
        max_tokens: Maximum number of tokens per chunk
        overlap: Number of overlapping tokens between consecutive chunks
        
    Returns:
        List of text chunks
    """
    chunker = TokenChunker(
        tokenizer="gpt2",  # Use GPT-2 tokenizer for consistent token counting
        chunk_size=max_tokens,  
        chunk_overlap=overlap,  
    )

    chunks = chunker.chunk(text)
    result = []

    # Extract text content from chunk objects
    for chunk in chunks:
        result.append(chunk.text)
    return result


def split_text_by_sentence(
    text: str,
    max_tokens: int = 30,
    overlap: int = 0,
    min_sentences_per_chunk: int = 1,
):
    """
    Split text into chunks based on sentence boundaries with token limits.
    
    Args:
        text: Input text to split
        max_tokens: Maximum number of tokens per chunk
        overlap: Number of overlapping tokens between consecutive chunks
        min_sentences_per_chunk: Minimum number of sentences required per chunk
        
    Returns:
        List of text chunks respecting sentence boundaries
    """
    chunker = SentenceChunker(
        tokenizer="gpt2",  # Use consistent tokenizer
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        min_sentences_per_chunk=min_sentences_per_chunk,
    )

    chunks = chunker.chunk(text)
    result = []

    # Extract text content from chunk objects
    for chunk in chunks:
        result.append(chunk.text)
    return result


def split_text_by_semantic(
    chunker: SemanticChunker,
    text: str,
):
    """
    Split text into semantically coherent chunks using provided semantic chunker.
    
    Args:
        chunker: Pre-configured SemanticChunker instance
        text: Input text to split
        
    Returns:
        List of semantically coherent text chunks
    """
    chunks = chunker.chunk(text)
    result = []

    # Extract text content from chunk objects
    for chunk in chunks:
        result.append(chunk.text)
    return result


def clear_parentheses(entity_text):
    """
    Clean entity text by removing parenthetical expressions.
    Used for entity normalization in knowledge graphs.
    
    Args:
        entity_text: Raw entity text that may contain parentheses
        
    Returns:
        Cleaned entity text with parentheses and their contents removed
    """
    # Remove leading/trailing whitespace
    cleaned = entity_text.strip()
    
    # Remove all parenthetical expressions (including nested ones)
    cleaned = re.sub(r"\s*\([^)]*\)", "", cleaned)
    return cleaned
