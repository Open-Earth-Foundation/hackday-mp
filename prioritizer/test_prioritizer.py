import os
import sys
from pathlib import Path
import requests

# --- Adjust Python Path to find 'prioritizer' and 'utils' ---
# This assumes test_prioritizer.py is in the root of the hackday_q1 project,
# alongside prioritizer.py and the utils directory.
# If your structure is different, you might need to adjust this.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# It's also good to ensure the parent of utils (if utils is a subdir) is in path
# For a flat structure where utils is a sibling of prioritizer.py and test_prioritizer.py, this might not be strictly needed
# but it's safer if there are nested imports within utils that expect a certain root.
# If prioritizer.py and utils are directly in PROJECT_ROOT, this is fine.

# --- Imports from project modules ---
try:
    from utils.reading_writing_data import read_city_inventory, read_actions
    # Functions from prioritizer.py that we want to test or use for testing
    from prioritizer import (
        initialize_openrouter_client_headers, 
        # get_openrouter_completion, # Called by rank_actions_with_llm
        # construct_comparison_prompt, # Called by rank_actions_with_llm
        rank_actions_with_llm,         # The function we are essentially testing
        load_action_pairs_from_file, # To get a test pair
        # load_models_from_file,       # To get a model for testing
        ACTION_PAIRS_FILE_PATH,      # Path to action_pairs.json
        # MODEL_LIST_FILE_PATH,        # Path to model_list.json (not needed for test anymore)
        # DEFAULT_MODEL_NAME           # Default model name handled by rank_actions_with_llm
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure that test_prioritizer.py is in the correct location (e.g., project root)")
    print("and that the necessary __init__.py files exist if you are treating them as packages.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Test Configuration ---
# The OPENROUTER_API_KEY should be set as an environment variable
# This script will fail if it's not set, as checked by initialize_openrouter_client_headers

def run_single_pair_test():
    """
    Loads necessary data, selects one pair and one model, and runs the prioritization.
    """
    print("=========================================")
    print(" Running Prioritizer Test Script ")
    print("=========================================")

    # 1. Initialize OpenRouter client headers (checks API key)
    try:
        print("\n--- Initializing OpenRouter Headers ---")
        openrouter_headers = initialize_openrouter_client_headers()
        print("OpenRouter headers initialized successfully.")
    except ValueError as e:
        print(f"Test Error: {e} - Cannot proceed without API key.")
        return

    # 2. Load all climate actions data
    print("\n--- Loading All Climate Actions Data ---")
    all_actions_data = read_actions() # Assuming this function converts to ActionID-keyed dict or we adapt
    if not all_actions_data:
        print("Test Error: Failed to load action data. Check 'data/climate_actions/merged.json' and utils.")
        return
    # Convert list to dict for quick lookup if not already done by read_actions
    # Based on prioritizer.py's load_all_actions_data, read_actions() returns a list.
    actions_dict = {action["ActionID"]: action for action in all_actions_data if action.get("ActionID")}
    if not actions_dict:
        print("Test Error: No actions with ActionID found after processing from read_actions().")
        return
    print(f"Successfully loaded {len(actions_dict)} actions into a dictionary.")


    # 3. Load action pairs for comparison
    print("\n--- Loading Action Pairs ---")
    action_pairs_list = load_action_pairs_from_file(ACTION_PAIRS_FILE_PATH)
    if not action_pairs_list:
        print(f"Test Error: No action pairs loaded from {ACTION_PAIRS_FILE_PATH}. Cannot proceed with test.")
        # Optionally use a hardcoded mock pair for testing if file loading is not the focus of the test
        # mock_test_pair = {"actionA_id": "YOUR_VALID_ACTION_A_ID", "actionB_id": "YOUR_VALID_ACTION_B_ID", "city_locode": "YOUR_VALID_LOCODE"}
        # if not all(mock_test_pair.values()): print("Mock test pair is not fully configured."); return
        # action_pairs_list = [mock_test_pair]
        # print("Using a hardcoded mock test pair as fallback.")
        return
    
    single_test_pair = action_pairs_list[0] # Take the first pair for testing
    print(f"Selected test pair: {single_test_pair}")

    # 4. Load models and select one for testing
    # Model selection is now handled by rank_actions_with_llm, which has a default.
    # We can pass a specific model_name to rank_actions_with_llm if we want to override the default for testing.
    # For now, we will use its default: "google/gemini-2.5-pro-preview"
    print(f"\n--- Model Selection ---")
    print(f"Test will use the default model in rank_actions_with_llm ('google/gemini-2.5-pro-preview') unless overridden.")
    # Example override: model_override_for_test = "specific/test-model"
    model_override_for_test = None # Set to a model name string to override

    # 5. Extract data for the single test pair
    action_a_id = single_test_pair.get("actionA_id")
    action_b_id = single_test_pair.get("actionB_id")
    city_locode = single_test_pair.get("city_locode")

    if not all([action_a_id, action_b_id, city_locode, isinstance(city_locode, str)]):
        print(f"  Test Error: Invalid data in selected test pair: {single_test_pair}. Check action IDs and city locode.")
        return

    action_a_data = actions_dict.get(action_a_id)
    action_b_data = actions_dict.get(action_b_id)

    if not action_a_data:
        print(f"  Test Error: ActionID '{action_a_id}' (Action A) not found in loaded actions.")
        return
    if not action_b_data:
        print(f"  Test Error: ActionID '{action_b_id}' (Action B) not found in loaded actions.")
        return

    try:
        print(f"  Reading city inventory for locode: {city_locode}...")
        city_context_data = read_city_inventory(city_locode) # From utils
        print(f"  Successfully read city data for {city_locode}.")
    except ValueError as e:
        print(f"  Test Error: Reading city inventory for '{city_locode}': {e}.")
        return
    except Exception as e:
        print(f"  Test Error: Unexpected error reading city inventory for '{city_locode}': {e}.")
        return

    # 6. Construct prompt and call LLM
    print("\n--- Calling LLM for Test Pair ---")

    test_result_payload = {
        "actionA_id": action_a_id,
        "actionB_id": action_b_id,
        "city_locode": city_locode,
        "winner": None,
        "error_detail": None,
        "llm_raw_response": None
    }

    try:
        # Call the refactored function directly
        rank_args = {
            "city_dict": city_context_data,
            "action_a_dict": action_a_data,
            "action_b_dict": action_b_data,
            "openrouter_headers": openrouter_headers
        }
        if model_override_for_test:
             rank_args["model_name"] = model_override_for_test
             print(f"  Overriding model for test: {model_override_for_test}")
         
        # rank_actions_with_llm handles prompt construction, API call, and basic response validation
        winner_action_id = rank_actions_with_llm(**rank_args)
        print(f"  Test LLM Chose: {winner_action_id}")
        test_result_payload["winner"] = winner_action_id
     
    # Catch exceptions that might be raised by rank_actions_with_llm or its sub-calls
    except ValueError as e:
        error_msg = f"Test Error during LLM call or validation: {e}"
        print(f"  {error_msg}")
        test_result_payload["winner"] = "Error: LLM ValueError"
        test_result_payload["error_detail"] = str(e)
    except requests.exceptions.RequestException as e: # Includes HTTPError
        error_msg = f"Test Error during API Request: {e}"
        print(f"  {error_msg}")
        test_result_payload["winner"] = "Error: API Request Failed"
        test_result_payload["error_detail"] = str(e)
    except KeyError as e:
        error_msg = f"Test Error processing LLM response keys: {e}"
        print(f"  {error_msg}")
        test_result_payload["winner"] = "Error: LLM Response Format Key Error"
        test_result_payload["error_detail"] = str(e)
    except Exception as e:
        error_msg = f"Unexpected Test Error during LLM call: {e}"
        print(f"  {error_msg}")
        test_result_payload["winner"] = "Error: LLM call failed"
        test_result_payload["error_detail"] = str(e)
    
    print("\n--- Single Pair Test Result ---")
    # Pretty print the result dictionary
    import json as json_printer # Alias to avoid conflict if json module is used differently elsewhere
    print(json_printer.dumps(test_result_payload, indent=2))
    
    print("=========================================")
    print(" Prioritizer Test Script Finished ")
    print("=========================================")

if __name__ == "__main__":
    run_single_pair_test() 