from langsmith import Client
from typing import List, Tuple

client = Client()

headers = [
    {"name": "meta-llama/llama-4-maverick:free", "has_structured_outputs": True},
    {"name": "meta-llama/llama-4-scout:free", "has_structured_outputs": True},
    {"name": "google/gemma-3-27b-it:free", "has_structured_outputs": True},
]


def mock_llm_comparison(inputs: dict, headers: list[dict] = headers) -> dict:

    # print(inputs)
    print(headers)

    locode = inputs["CityLocode"]
    action_a = inputs["ActionA"]
    action_b = inputs["ActionB"]

    # LLM logic here
    llm_response = action_b

    return {"PreferredAction": llm_response}


# Define the application logic you want to evaluate inside a target function
# The SDK will automatically send the inputs from the dataset to your target function
# def target(inputs: dict) -> dict:
#     # inputs is the input of the test dataset
#     # llm compare (action a aciton b city)
#     # return action b
#     # Your model logic here, for example:
#     predicted_action = "icare_0003"  # Replace with actual prediction logic
#     return {"PreferredAction": predicted_action}


def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    # outputs is the output of the 'target' function
    # reference_outputs is the output of the test dataset

    # Compare predicted PreferredAction to the reference PreferredAction
    return outputs.get("PreferredAction") == reference_outputs.get("PreferredAction")


def evaluation_function(
    dataset_name: str = "ds-expert-comparisons",
    experiment_prefix: str = "first-eval-in-langsmith",
):
    """
    This function loads the test data from the dataset and runs the evaluation.
    It will take the 'inputs' from the test dataset and pass them to the 'target' function.
    It will then compare the output of the 'target' function to the 'reference_outputs' from the test dataset.
    """

    # After running the evaluation, a link will be provided to view the results in langsmith
    experiment_results = client.evaluate(
        mock_llm_comparison,
        data=dataset_name,
        evaluators=[
            correctness_evaluator,  # type: ignore
        ],
        experiment_prefix=experiment_prefix,
    )  # type: ignore


if __name__ == "__main__":
    # Load the test data
    evaluation_function(
        dataset_name="ds-expert-comparisons",
        experiment_prefix="first-eval-in-langsmith",
    )
