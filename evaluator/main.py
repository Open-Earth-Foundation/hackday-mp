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

MODEL_NAME = "google/gemini-flash-1.5"


# Wrapper function to pass in a model name
def target_with_model(inputs: dict) -> dict:
    return rank_actions_with_llm(inputs, model_name=MODEL_NAME)


# def mock_llm_comparison(inputs: dict, headers: list[dict] = headers) -> dict:

#     # print(inputs)
#     print(headers)

#     locode = inputs["CityLocode"]
#     action_a = inputs["ActionA"]
#     action_b = inputs["ActionB"]

#     # LLM logic here
#     llm_response = action_b

#     return {"PreferredAction": llm_response}


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
    """

    # After running the evaluation, a link will be provided to view the results in langsmith
    experiment_results = client.evaluate(
        target_with_model,
        data=dataset_name,
        evaluators=[
            correctness_evaluator,  # type: ignore
        ],
        experiment_prefix=experiment_prefix,
    )  # type: ignore


if __name__ == "__main__":
    # Load the test data
    evaluation_function(
        experiment_prefix=MODEL_NAME, dataset_name="ds-expert-comparisons"
    )
