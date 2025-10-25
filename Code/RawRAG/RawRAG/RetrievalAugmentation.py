import logging
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer
import tiktoken
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from transformers import AutoModel
import os
import pickle
import faiss
from tqdm import tqdm

from RawRAG.EmbeddingModels import BaseEmbeddingModel, SBertEmbeddingModel, JinaEmbeddingModel
from RawRAG.QAModels import BaseQAModel
from RawRAG.utils import (
    split_text_by_semantic,
    split_text_by_sentence,
    split_text_by_token,
    clear_parentheses,
)
from .RGP import RGP, TimeCausalMask

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalConfig:
    """Configuration class for RAG system parameters."""
    
    def __init__(
        self,
        qa_model: Optional[BaseQAModel] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        chunk_size: int = 100,
        chunk_overlap: int = 10,
        tokenizer=None,
        iter: int = 5,
        prob: float = 0.9,
        batch_size: int = 1,
        filter_chunk: bool = False,
        layer: int = 14,
    ):
        """
        Initialize RAG configuration.
        
        Args:
            qa_model: Question answering model instance
            embedding_model: Text embedding model instance
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
            tokenizer: Tokenizer for text processing (defaults to tiktoken)
            iter: Number of RGP random walk iterations
            prob: Random walk restart probability
            batch_size: Batch size for processing
            filter_chunk: Whether to apply attention-based chunk filtering
            layer: Model layer for attention extraction
        """
        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")
        self.qa_model = qa_model
        
        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        self.embedding_model = embedding_model

        # Text processing parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        
        # RGP algorithm parameters
        self.iter = iter  # Random walk iterations
        self.prob = prob  # Restart probability
        
        # Processing parameters
        self.batch_size = batch_size
        self.filter_chunk = filter_chunk  # Attention-based filtering
        self.layer = layer  # Model layer for attention

class RetrievalAugmentation:
    """
    Main RAG system class that combines retrieval and generation.
    
    This system performs entity-aware document retrieval using graph propagation
    followed by question answering on the retrieved context.
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        Initialize the RAG system.
        
        Args:
            config: Configuration object with model and processing parameters
        """
        if config is None:
            config = RetrievalConfig()

        self.config = config
        self.qa_model = config.qa_model
        self.embedding_model = config.embedding_model
        self.tokenizer = config.tokenizer
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.iter = config.iter
        self.prob = config.prob
        self.batch_size = config.batch_size
        self.filter_chunk = config.filter_chunk

        # Storage for processed corpus data
        self.text_chunks = []  # Processed text chunks from documents
        self.embeddings = []  # Embedding vectors for chunks
        self.corpus = []  # Original corpus with metadata
        self.layer = config.layer  # Model layer for attention extraction

    def build_entity_list(self) -> Tuple[List[str], Dict[str, int]]:
        """
        Build list of entities that appear in multiple documents for graph construction.
        
        Only entities that appear in multiple documents are included to ensure
        meaningful connections in the heterogeneous graph.
        
        Returns:
            Tuple of (entity_list, entity_str_to_idx):
            - entity_list: List of entity strings that appear in multiple documents
            - entity_str_to_idx: Mapping from entity string to index
        """
        entity_list = []
        entity_set = set()
        entity_to_idx = {}

        # First pass: collect all entities and their document associations
        for corpus_idx, item in enumerate(self.corpus):
            entities = item["entities"]
            for entity in entities:
                entity_text = entity["text"]
                # Normalize entity text (remove parentheses, lowercase)
                entity = clear_parentheses(entity_text)
                entity = entity.lower()
                
                if entity not in entity_set:
                    entity_set.add(entity)
                    entity_to_idx[entity] = len(entity_list)
                    entity_list.append(
                        {
                            "entity": entity,
                            "corpus_idx": [corpus_idx],  # Documents containing this entity
                        }
                    )
                else:
                    # Add this document to the entity's document list
                    entity_list[entity_to_idx[entity]]["corpus_idx"].append(corpus_idx)

        # Second pass: keep only entities that appear in multiple documents
        final_entity_str_to_idx = {}
        final_entity_list = []
        cnt = 0
        for entity in entity_list:
            if len(entity["corpus_idx"]) > 1:  # Multi-document entities only
                final_entity_list.append(entity["entity"])
                final_entity_str_to_idx[entity["entity"]] = cnt
                cnt += 1

        return final_entity_list, final_entity_str_to_idx

    def build_embedding_graph(self) -> np.ndarray:
        """
        Build similarity graph between text chunks based on embedding cosine similarity.
        
        Creates an undirected graph where edge weights represent exponential of
        cosine similarity between chunk embeddings.
        
        Returns:
            Symmetric similarity matrix between chunks with -inf on diagonal
            
        Raises:
            ValueError: If no documents have been processed yet
        """
        if self.text_chunks is None:
            raise ValueError("No documents have been added yet.")
        
        chunk_num = len(self.text_chunks)
        
        # Create symmetric similarity matrix
        chunk_to_chunk = np.zeros((chunk_num, chunk_num))
        for i in range(chunk_num):
            for j in range(i + 1, chunk_num):
                # Use exponential of cosine similarity as edge weight
                similarity = np.exp(self._cal_cosine_similarity(self.embeddings[i], self.embeddings[j]))
                chunk_to_chunk[i, j] = similarity
                chunk_to_chunk[j, i] = similarity
            # Set diagonal to -inf to prevent self-loops in random walk
            chunk_to_chunk[i, i] = -np.inf
        return chunk_to_chunk
    
    def build_entity_graph(self) -> np.ndarray:
        """
        Build bipartite graph between documents and entities.
        
        Creates connections between documents and the entities they contain.
        Only includes entities that appear in multiple documents.
        
        Returns:
            Binary adjacency matrix [corpus_size x entity_size] where 1 indicates
            that a document contains an entity, -inf indicates no connection
        """
        corpus_size, entity_size = len(self.corpus), len(self.entity_list)
        # Initialize with -inf (no connections)
        entity_graph = np.ones((corpus_size, entity_size)) * -np.inf

        # Fill in actual entity-document connections
        for corpus_idx, item in enumerate(self.corpus):
            entities = item["entities"]
            for entity in entities:
                entity_text = entity["text"]
                # Normalize entity text to match entity list
                entity = clear_parentheses(entity_text)
                entity = entity.lower()
                # If entity is in our filtered list, mark connection
                if entity in self.entity_str_to_idx:
                    entity_graph[corpus_idx, self.entity_str_to_idx[entity]] = 1

        return entity_graph
    
    def _cal_cosine_similarity(self, a, b):
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve_documents(
        self,
        queries: List[str],
        top_k_final: int = 10,
        questions: List[str] = None,
        question_entities: List[List[Dict]] = None,
        attn_topk: int = 25,
        batch_no: int = 0,
    ) -> List[List[Dict]]:
        batch_size = len(queries)

        all_semantic_chunk_question_probs = []
        all_entity_chunk_question_probs = []
        for query_idx in range(batch_size):
            # Get entity list for the current question
            question_entity_list = question_entities[query_idx]

            question_embedding = self.embedding_model.create_embedding(questions[query_idx], "query")
            
            # Calculate similarity between question and all chunks in corpus as n*1 part
            semantic_chunk_question_probs = np.ones((len(self.text_chunks), 1)) * -np.inf
            for idx, embedding in enumerate(self.embeddings):
                semantic_chunk_question_probs[idx] = np.exp(self._cal_cosine_similarity(question_embedding, embedding))
            
            # Calculate similarity between question and all entities in corpus as m*1 part (excluding isolated entities)
            entity_chunk_question_probs = np.ones((len(self.entity_list), 1)) * -np.inf
            for entity in question_entity_list:
                entity_text = entity.get("text", "")
                entity_text = clear_parentheses(entity_text)
                entity_text = entity_text.lower()
                if entity_text in self.entity_str_to_idx:
                    entity_chunk_question_probs[self.entity_str_to_idx[entity_text], :] = 1

            all_semantic_chunk_question_probs.append(semantic_chunk_question_probs)
            all_entity_chunk_question_probs.append(entity_chunk_question_probs)

        all_semantic_chunk_question_probs = np.array(all_semantic_chunk_question_probs).reshape(batch_size, -1)
        all_entity_chunk_question_probs = np.array(all_entity_chunk_question_probs).reshape(batch_size, -1)


        # 4. Prepare RGP input
        rgp = RGP(device="cuda", iter=self.iter, factor=20, prob=self.prob)
        semantic_matrix = self.convert_to_tensor(self.embedding_graph).unsqueeze(0).expand(batch_size, -1, -1)
        entity_matrix = self.convert_to_tensor(self.entity_graph).unsqueeze(0).expand(batch_size, -1, -1)

        # 5. Calculate attention scores using RGP
        probs, _, path = rgp.forward(
            semantic_matrix,
            entity_matrix,
            chunk_question_probs=all_semantic_chunk_question_probs,
            entity_question_probs=all_entity_chunk_question_probs,
            use_analytic=False,
            use_sparse=True,
        )

        probs = probs[..., 1:1 + len(self.corpus)]

        # 6. Process results
        probs = probs.cpu()  # [batch_size, 1, chunk_size]
        batch_results = []

        for batch_idx in range(batch_size):
            final_chunk = []
            prob = probs[batch_idx].reshape(-1)  # [chunk_size]

            # Sort by probability
            sorted_prob, sorted_indices = torch.sort(prob, descending=True)
            
            # Only process chunks retrieved by the current query
            sorted_prob = sorted_prob.numpy()
            sorted_indices = sorted_indices.numpy()
            for prob, idx in zip(sorted_prob[:top_k_final], sorted_indices[:top_k_final]):
                chunk_id = f"chunk_{idx}"
                if prob > 0:
                    final_chunk.append({
                        "title": self.corpus[idx]["title"],
                        "content": self.text_chunks[idx],
                        "similarity": prob,
                        "chunk_id": chunk_id,
                    })

            filtered_chunks = final_chunk

            batch_results.append(filtered_chunks)

        return batch_results

    def convert_to_tensor(self, matrix) -> torch.Tensor:
        return torch.from_numpy(matrix).float().cuda()

    def answer_questions_corpus(self, questions: List[str], top_k_final: int = 10, question_entities: List[List[Dict]] = None, attn_topk: int = 25, batch_no: int = 0):
        """Answer a batch of questions while maintaining document independence"""
        # Initialize answers list at the beginning
        answers = ["No relevant information found."] * len(questions)

        # Process each question with its corresponding document to get contexts
        all_contexts = []

        # Use multi-center retrieval to get relevant chunks
        retrieved_chunks = self.retrieve_documents(
            queries=questions,
            top_k_final=top_k_final,
            questions=questions,
            question_entities=question_entities,
            attn_topk=attn_topk,
            batch_no=batch_no,
        )

        # Extract and combine context from retrieved chunks
        for retrieved_chunk in retrieved_chunks:
            contexts = [chunk["content"] for chunk in retrieved_chunk]
            context = "\n\n".join(contexts)
            # Store context and question for batch processing
            all_contexts.append(context)

        # Get answers from QA model in batch
        batch_answers = self.qa_model.answer_questions_batch(
            contexts=all_contexts,
            questions=questions,
            max_tokens=1000,
            batch_size=self.batch_size,
        )

        # Place answers in their original positions
        for idx, answer in zip(range(len(questions)), batch_answers):
            answers[idx] = answer

        return answers, all_contexts, retrieved_chunks

    def add_documents_corpus(self, corpus, questions):
        # 1. Create embeddings
        corpus_text = [item["text"] for item in corpus]
        embeddings = []
        for item in corpus_text:
            embedding = self.embedding_model.create_embedding(item)
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        # 2. Save results
        self.text_chunks = corpus_text
        self.corpus = corpus
        self.questions = questions
        self.embeddings = embeddings

        # 3. Build entity list
        self.entity_list, self.entity_str_to_idx = self.build_entity_list()

        # 4. Build graphs
        self.embedding_graph = self.build_embedding_graph()
        self.entity_graph = self.build_entity_graph()

    def save_corpus(self, save_dir: str = "results/saved_graph"):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "text_chunks.pkl"), "wb") as f:
            pickle.dump(self.text_chunks, f, protocol=5)
        with open(os.path.join(save_dir, "corpus.pkl"), "wb") as f:
            pickle.dump(self.corpus, f, protocol=5)
        with open(os.path.join(save_dir, "embeddings.pkl"), "wb") as f:
            pickle.dump(self.embeddings, f, protocol=5)
        with open(os.path.join(save_dir, "entity_list.pkl"), "wb") as f:
            pickle.dump(self.entity_list, f, protocol=5)
        with open(os.path.join(save_dir, "entity_str_to_idx.pkl"), "wb") as f:
            pickle.dump(self.entity_str_to_idx, f, protocol=5)
        # Save chunk_to_landmark
        torch.save(self.embedding_graph, os.path.join(save_dir, "embedding_graph.pt"), pickle_protocol=5)
        # Save chunk_to_chunk
        torch.save(self.entity_graph, os.path.join(save_dir, "entity_graph.pt"), pickle_protocol=5)
        logging.info(f"Successfully saved corpus to {save_dir}")
    
    def load_corpus(self, save_dir: str = "results/saved_graph"):
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory {save_dir} does not exist")

        try:
            # Load text chunks
            with open(os.path.join(save_dir, "text_chunks.pkl"), "rb") as f:
                self.text_chunks = pickle.load(f)
            # Load entity list
            with open(os.path.join(save_dir, "corpus.pkl"), "rb") as f:
                self.corpus = pickle.load(f)
            # Load embeddings
            with open(os.path.join(save_dir, "embeddings.pkl"), "rb") as f:
                self.embeddings = pickle.load(f)
            # Load entity list
            with open(os.path.join(save_dir, "entity_list.pkl"), "rb") as f:
                self.entity_list = pickle.load(f)
            # Load entity str to idx
            with open(os.path.join(save_dir, "entity_str_to_idx.pkl"), "rb") as f:
                self.entity_str_to_idx = pickle.load(f)
            # Load embedding graph
            self.embedding_graph = torch.load(os.path.join(save_dir, "embedding_graph.pt"))
            # Load entity graph
            self.entity_graph = torch.load(os.path.join(save_dir, "entity_graph.pt"))
            logging.info(f"Successfully loaded corpus from {save_dir}")
        except Exception as e:
            raise Exception(f"Error loading corpus from {save_dir}: {str(e)}")
