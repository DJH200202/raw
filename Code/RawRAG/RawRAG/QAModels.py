import logging
import os
import gc
from typing import List

from openai import OpenAI
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


class BaseQAModel(ABC):
    """Abstract base class for all question answering models."""
    
    @abstractmethod
    def answer_question(self, context, question, max_tokens):
        """
        Answer a single question given context.
        
        Args:
            context: Retrieved context text
            question: Question to answer
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer string
        """
        pass

    @abstractmethod
    def answer_questions_batch(self, contexts, questions, max_tokens):
        """
        Answer multiple questions in batch.
        
        Args:
            contexts: List of context texts
            questions: List of questions
            max_tokens: Maximum tokens to generate per answer
            
        Returns:
            List of generated answers
        """
        pass


class Llama3QAModel(BaseQAModel):
    """Local Llama3 model for question answering using HuggingFace transformers."""
    
    def __init__(self, model_name="/home/ubuntu/model/Meta-Llama-3-8B-Instruct"):
        """
        Initialize Llama3 model with text generation pipeline.
        
        Args:
            model_name: Path to local Llama3 model directory
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Create text generation pipeline for QA
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.qa_pipeline.tokenizer.pad_token = self.qa_pipeline.tokenizer.eos_token
        self.qa_pipeline.tokenizer.padding_side = "left"

    def answer_question(self, context, question, max_tokens=1000):
        """
        Generate answer for a single question using Llama3.
        
        Args:
            context: Retrieved context text
            question: Question to answer
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer string
        """
        # Format input as chat conversation
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions accurately based on the given context. Only give me the answer and do not output any other words.",
            },
            {
                "role": "user",
                "content": f"Given Context: {context}\nQuestion: {question}\nPlease provide a clear and concise answer and do not output any other words.",
            },
        ]
        # Apply chat template for proper formatting
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate answer with controlled parameters
        outputs = self.qa_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,  # Moderate creativity
            top_k=50,  # Limit vocabulary choices
            top_p=0.95,  # Nucleus sampling
        )

        # Extract only the new generated text (remove prompt)
        answer = outputs[0]["generated_text"][len(prompt) :]
        return answer

    def answer_questions_batch(
        self,
        contexts: List[str],
        questions: List[str],
        max_tokens: int,
        batch_size: int,
    ) -> List[str]:
        """
        Generate answers for multiple questions in batch for efficiency.
        
        Args:
            contexts: List of retrieved context texts
            questions: List of questions to answer
            max_tokens: Maximum tokens to generate per answer
            batch_size: Number of examples to process together
            
        Returns:
            List of generated answers
            
        Raises:
            ValueError: If contexts and questions lists have different lengths
        """
        if len(contexts) != len(questions):
            raise ValueError("Number of contexts must match number of questions")

        # Prepare all prompts using chat templates
        prompts = []
        for context, question in zip(contexts, questions):
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions accurately based on the given context. Only give me the answer and do not output any other words.",
                },
                {
                    "role": "user",
                    "content": f"Given Context: {context}\nQuestion: {question}\nPlease provide a clear and concise answer and do not output any other words.",
                },
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        # Generate answers in batch with consistent parameters
        outputs = self.qa_pipeline(
            prompts,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.2,  # Lower temperature for more consistent answers
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            padding=True,
            return_full_text=False,  # Only return generated text
            batch_size=batch_size,
        )

        # Extract answers from pipeline output
        answers = []
        for output in outputs:
            answer = output[0]["generated_text"]
            answers.append(answer)

        return answers

    def get_attention_scores(self, question="", context="", layer_indices=None):
        """
        Extract attention weights from Llama3 model for attention-based filtering.
        
        Args:
            question: Input question
            context: Context text
            layer_indices: Specific layer indices to extract (if None, returns all layers)
            
        Returns:
            Attention tensor of shape [num_layers, num_heads, seq_len, seq_len]
        """
        model = self.qa_pipeline.model
        
        context_truncated = context

        # Format input for attention extraction
        messages = [
            {
                "role": "user",
                "content": f"Context: {context_truncated}\nQuestion: {question}",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        # Forward pass with attention output
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Extract and move attention matrices to CPU for processing
        all_attention_matrices = [attn.cpu() for attn in outputs.attentions]

        # Clean up GPU memory
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        # Stack attention matrices and reshape
        attention_tensor = torch.stack(all_attention_matrices)
        
        # Remove batch dimension (assuming batch_size=1)
        attention_tensor = attention_tensor[:, 0, :, :, :]
        
        # Filter specific layers if requested
        if layer_indices is not None:
            attention_tensor = attention_tensor[layer_indices]

        return attention_tensor
    

class GPTQAModel(BaseQAModel):
    """
    Hybrid QA model using OpenAI GPT for answering and local Llama3 for attention extraction.
    This combines the quality of GPT models with local attention analysis capabilities.
    """
    
    def __init__(self, model="gpt-3.5-turbo", layer=14):
        """
        Initialize hybrid GPT + Llama3 model.
        
        Args:
            model: OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4o")
            layer: Number of Llama3 layers to keep for attention extraction
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Internal method to call OpenAI API with retry logic.
        
        Args:
            context: Retrieved context text
            question: Question to answer
            max_tokens: Maximum tokens to generate (not used for GPT API)
            stop_sequence: Stop sequences (not used)
            
        Returns:
            Generated answer from OpenAI API
            
        Raises:
            Exception: If API call fails after retries
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise and precise assistant. You answer only using the provided context.\n"
                            "Always return the most direct and minimal answer possible.\n"  
                            "IMPORTANT:\n"
                            "- Only output the answer content directly.\n"
                            "- INCLUDE ALL NECESSARY DESCRIPTIVE ELEMENTS from the context that identify the answer (e.g., 'the State House in Augusta' not just 'Augusta').\n"
                            "- Use EXACT wording from the context when it's the correct answer.\n"
                            "- Include ALL words needed for the answer to be complete and accurate.\n"
                            "- Do NOT add words that aren't necessary to answer the question.\n"
                        )  
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context: {context}\n"
                            f"Question: {question}\n"
                            "Only return the final answer. No extra words.\n"
                            "Answer:"
                        )
                    },
                ],
                temperature=0,  # Deterministic responses
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Answer a single question using OpenAI GPT model.
        
        Args:
            context: Retrieved context text
            question: Question to answer
            max_tokens: Maximum tokens (not used for OpenAI API)
            stop_sequence: Stop sequences (not used)
            
        Returns:
            Generated answer string or error object
        """
        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_questions_batch(
        self,
        contexts: List[str],
        questions: List[str],
        max_tokens: int = 150,
        stop_sequence=None,
        batch_size: int = 4,
    ) -> List[str]:
        """
        Answer multiple questions in parallel using OpenAI API.
        
        Args:
            contexts: List of retrieved context texts
            questions: List of questions to answer
            max_tokens: Maximum tokens (not used for OpenAI API)
            stop_sequence: Stop sequences (not used)
            batch_size: Number of concurrent API calls
            
        Returns:
            List of generated answers
            
        Raises:
            ValueError: If contexts and questions lists have different lengths
        """
        if len(contexts) != len(questions):
            raise ValueError("Number of contexts must match number of questions")

        answers = [None] * len(contexts)
        
        def process_item(idx, context, question):
            """Process a single question-context pair."""
            try:
                answer = self._attempt_answer_question(
                    context,
                    question,
                    max_tokens=max_tokens,
                    stop_sequence=stop_sequence,
                )
            except Exception as e:
                print(f"Error in process_item for question {idx}: {e}")
                answer = f"Error: {str(e)}"
            return idx, answer
        
        # Use ThreadPoolExecutor for concurrent API calls
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(process_item, i, context, question) 
                for i, (context, question) in enumerate(zip(contexts, questions))
            ]
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                idx, answer = future.result()
                answers[idx] = answer
        
        return answers

