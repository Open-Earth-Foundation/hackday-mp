"""
This script is used to evaluate the correctness of the prioritizer.

To run the script, use the following command from root directory:
python -m evaluator.main

Set the MODEL_NAME to the model you want to evaluate.
The langsmith experiment will be saved with this model name.
"""

from langsmith import Client
from typing import List, Tuple
from prioritizer.prioritizer import rank_actions_with_llm

client = Client()

# MODEL_NAME = "google/gemini-flash-1.5"
# MODEL_NAME = "meta-llama/llama-4-maverick:free"
#MODEL_NAME = "qwen/qwen3-30b-a3b:free"
#MODEL_NAME = "google/gemini-2.5-flash-preview"
MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"

# Wrapper function to pass in a model name
def target_with_model(inputs: dict) -> dict:
    return rank_actions_with_llm(inputs, model_name=MODEL_NAME)



def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    # outputs is the output of the 'target' function
    # reference_outputs is the output of the test dataset

    # Compare predicted PreferredAction to the reference PreferredAction
    return outputs.get("PreferredAction") == reference_outputs.get("PreferredAction")


def evaluation_function(experiment_prefix: str, dataset_name: str):
    """
    This function loads the test data from the dataset and runs the evaluation.
    It will take the 'inputs' from the test dataset and pass them to the 'target' function.
    It will then compare the output of the 'target' function to the 'reference_outputs' from the test dataset.
    Finally, it prints a summary of the evaluation results including error counts.
    """

    # After running the evaluation, a link will be provided to view the results in langsmith
    experiment_results = client.evaluate(
        target_with_model,
        data=dataset_name,
        evaluators=[
            correctness_evaluator,  # type: ignore
        ],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
    )  # type: ignore

    print("\n--- Evaluation Summary ---")
    if experiment_results and hasattr(experiment_results, 'summary'):
        summary = experiment_results.summary
        print(f"Experiment Name: {experiment_results.experiment_name}")
        print(f"Total Inputs: {summary.total_inputs}")
        print(f"Total Runs: {summary.total_runs}") # Should generally match total_inputs
        
        correctness_key = 'correctness_evaluator' # Assuming this is the key langsmith uses or the name of your evaluator
        if hasattr(summary, 'feedback_stats') and summary.feedback_stats and correctness_key in summary.feedback_stats:
            correctness_stats = summary.feedback_stats[correctness_key]
            # feedback_stats can be None if no feedback was recorded
            if correctness_stats:

                passed_count = correctness_stats.get('n_passed', 0) # Default to 0 if key not found
                failed_count = correctness_stats.get('n_failed', 0)

                print("Feedback for 'correctness_evaluator':")
                for key, value in correctness_stats.items():
                    print(f"  - {key}: {value}")

                # Let's assume 'n_failed' represents evaluation failures (e.g., "incorrect" outputs)
                # and 'error_count' represents runtime errors in the target function.
                evaluation_failures = failed_count 
                print(f"Evaluation Failures (Incorrect Outputs): {evaluation_failures}")

        else:
            print("No feedback stats available for 'correctness_evaluator'.")

        # Errors during the execution of the target function
        runtime_errors = summary.error_count if hasattr(summary, 'error_count') else 0
        print(f"Runtime Errors (in target function): {runtime_errors}")
        
        # Latency if available
        if hasattr(summary, 'latency_avg') and summary.latency_avg is not None:
            print(f"Average Latency: {summary.latency_avg:.2f}s")
        
        print(f"LangSmith Experiment URL: {experiment_results.url}")

    else:
        print("Experiment results or summary not available.")


if __name__ == "__main__":
    # Load the test data
    evaluation_function(
        experiment_prefix=MODEL_NAME, dataset_name="ds-expert-comparisons"
    )
