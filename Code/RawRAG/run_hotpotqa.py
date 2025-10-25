import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from RawRAG import (
    RetrievalAugmentation,
    RetrievalConfig,
    Llama3QAModel,
    JinaEmbeddingModel,
    SBertEmbeddingModel,
    GPTQAModel,
    OpenAIEmbeddingModel,
)
from metrics import normalize_answer


def seed_everything(seed: int):
    """Set random seeds for reproducibility across all random number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args():
    """Parse command line arguments for RAG system configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=1, help="Number of RGP iterations")
    parser.add_argument("--prob", type=float, default=0.9, help="Random walk probability")
    parser.add_argument("--filter", type=bool, default=False, help="Whether to filter chunks using attention")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top chunks to retrieve")
    parser.add_argument("--attn_topk", type=int, default=100, help="Number of top attention chunks")
    return parser.parse_args()


def run_hotpotqa(
    ra_system,
    output_dir="results/hotpotqa_eval",
    batch_size=8,
    top_k_final=10,
    iter=2,
    prob=0.9,
    attn_topk=25,
    filter=True,
):
    """
    Run RAG system evaluation on HotpotQA dataset.
    
    Args:
        ra_system: Initialized RetrievalAugmentation system
        output_dir: Directory to save results
        batch_size: Number of questions to process in each batch
        top_k_final: Number of top documents to retrieve
        iter: Number of RGP iterations
        prob: Random walk probability for RGP
        attn_topk: Number of top attention chunks for filtering
        filter: Whether to apply attention-based filtering
    
    Returns:
        Path to the saved results file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load HotpotQA validation dataset
    with open(os.path.join("data", "hippo", "hotpotqa.json"), "r") as f:
        ds = json.load(f)
    
    # Load pre-extracted question entities for entity-aware retrieval
    with open(
        os.path.join("results", "question_entities", "hotpotqa_question_entities.json"),
        "r",
    ) as f:
        question = json.load(f)

    results = []

    graph_path = "results/saved_graph/hotpot_corpus_entities_graph"
    # Load or create corpus with entity graphs for retrieval
    if not os.path.exists(graph_path):
        os.makedirs(graph_path, exist_ok=True)
        corpus = []
        with open(
            os.path.join("results", "corpus_entities", "hotpotqa_corpus_entities.json"),
            "r",
        ) as f:
            corpus = json.load(f)
        # Build and save entity-augmented corpus and graphs
        ra_system.add_documents_corpus(corpus, question)
        ra_system.save_corpus(save_dir=graph_path)
    else:
        # Load pre-built corpus and graphs
        ra_system.load_corpus(save_dir=graph_path)

    all_question_entities = [item["entities"] for item in question]

    # Process dataset examples in batches for efficiency
    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds[i : i + batch_size]
        batch_question_entities = all_question_entities[i : i + batch_size]
        batch_contexts = []
        batch_questions = []

        # Prepare batch data by combining context paragraphs
        for example in batch:
            # Extract paragraphs from context
            contexts = []
            for title, sentences in example["context"]:
                paragraph = f"{title}\n{' '.join(sentences)}"
                contexts.append(paragraph)

            full_context = "\n\n".join(contexts)
            batch_contexts.append(full_context)
            batch_questions.append(example["question"])

        # Use RAG system to retrieve relevant contexts and generate predictions
        predictions, retrieved_contexts, retrieved_chunks = (
            ra_system.answer_questions_corpus(
                questions=batch_questions,
                top_k_final=top_k_final,
                question_entities=batch_question_entities,
                attn_topk=attn_topk,
                batch_no=i,
            )
        )

        # Store results for evaluation
        for example, context, prediction, chunk in zip(
            batch, retrieved_contexts, predictions, retrieved_chunks
        ):
            result = {
                "question": example["question"],
                "context": context,
                "answers": [example["answer"]],
                "pred": prediction,
            }
            results.append(result)

    # Create unique output directory based on hyperparameters
    last_dir = f"iter_{iter}_prob_{prob}_top{top_k_final}"
    if filter:
        last_dir += f"_filter{attn_topk}"
    base_dir = os.path.join(output_dir, last_dir)
    dir_index = 0
    target_dir = base_dir
    while os.path.exists(target_dir):
        dir_index += 1
        target_dir = f"{base_dir}_{dir_index}"

    # Save results in JSONL format
    os.makedirs(target_dir)
    output_file = os.path.join(target_dir, "hotpot_validation_results.jsonl")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Results saved to: {output_file}")
    return output_file


def main():
    """Main execution function that sets up and runs the RAG evaluation."""
    args = parse_args()
    # Set seed for reproducible results
    seed_everything(224)

    batch_size = 4

    # Initialize RAG system with specified configuration
    config = RetrievalConfig(
        qa_model=Llama3QAModel(),  # Alternative: use local Llama model
        # qa_model=GPTQAModel(model="gpt-4o"),  # Using GPT-4o for question answering
        embedding_model=JinaEmbeddingModel(),  # Jina embeddings for document representation
        chunk_size=512,  # Size of text chunks for processing
        chunk_overlap=128,  # Overlap between consecutive chunks
        iter=args.iter,  # Number of RGP random walk iterations
        prob=args.prob,  # Random walk restart probability
        batch_size=batch_size,  # Batch size for processing
        filter_chunk=args.filter,  # Whether to apply attention filtering
    )
    ra_system = RetrievalAugmentation(config=config)

    # Run evaluation on HotpotQA dataset
    run_hotpotqa(
        ra_system,
        batch_size=batch_size,
        top_k_final=args.top_k,
        iter=args.iter,
        prob=args.prob,
        attn_topk=args.attn_topk,
        filter=args.filter,
    )



if __name__ == "__main__":
    main()
