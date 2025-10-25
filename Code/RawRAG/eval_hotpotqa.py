
import os
import json
import argparse
import numpy as np
from metrics import exact_match_score, qa_f1_score

def parse_args():
    """Parse command line arguments for evaluation configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="results/hotpotqa_eval/hotpotqa_validation_results.jsonl", help="Path to the results file")
    parser.add_argument("--output_file", type=str, default="results/hotpotqa_eval/hotpotqa_eval_results.json", help="Path to save evaluation results")
    return parser.parse_args()

def scorer(predictions, answers, all_classes=None):
    """
    Calculate evaluation scores for predictions against ground truth answers.
    
    Args:
        predictions: List of predicted answers
        answers: List of ground truth answers (each can have multiple correct answers)
        all_classes: Optional classification classes for specialized scoring
    
    Returns:
        Dictionary containing averaged exact match and F1 scores as percentages
    """
    total_scores = {}
    
    # Iterate through each prediction-answer pair
    for prediction, ground_truths in zip(predictions, answers):
        scores = {"em": 0.0, "f1": 0.0}
        
        # For each possible ground truth answer, take the maximum score
        for ground_truth in ground_truths:
            scores["em"] = max(
                scores["em"],
                exact_match_score(prediction, ground_truth, all_classes=all_classes)
            )
            scores["f1"] = max(
                scores["f1"],
                qa_f1_score(prediction, ground_truth, all_classes=all_classes)
            )
        
        # Accumulate scores across all examples
        for k, v in scores.items():
            if k not in total_scores:
                total_scores[k] = 0.0
            total_scores[k] += v
    
    # Convert to percentages and round to 2 decimal places
    for k in total_scores:
        total_scores[k] = round(100 * total_scores[k] / len(predictions), 2)
    
    return total_scores

def main():
    """Main evaluation function that loads results, calculates scores, and saves output."""
    args = parse_args()
    
    # Load predictions and ground truth answers from results file
    predictions = []
    answers = []
    all_classes = None
    
    with open(args.results_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Extract prediction (try different possible field names)
                predictions.append(data.get("pred", data.get("prediction", "")))
                # Extract answers (ensure it's a list for multiple possible correct answers)
                answers.append(data.get("answers", [data.get("ground_truth", "")]))
                # Extract classification classes if available
                if "all_classes" in data:
                    all_classes = data["all_classes"]
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    
    # Calculate evaluation scores
    scores = scorer(predictions, answers, all_classes)
    
    # Save evaluation results to output file
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump({"hotpotqa": scores}, f, ensure_ascii=False, indent=4)
    
    # Display results to console
    print(f"dataset hotpotqa scores {scores}")

if __name__ == "__main__":
    main()
