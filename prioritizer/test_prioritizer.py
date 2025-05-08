import os
import sys
from pathlib import Path
import requests
from unittest.mock import patch

# --- Adjust Python Path to find 'prioritizer' and 'utils' ---
# Get the directory containing this test file ('prioritizer')
TEST_DIR = Path(__file__).resolve().parent
# Get the parent directory (project root, e.g., 'hackday-mp')
PROJECT_ROOT = TEST_DIR.parent 
# Add the project root to the beginning of sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports from project modules ---
# Now imports should work relative to PROJECT_ROOT
# Assuming prioritizer.py is in the 'prioritizer' subdirectory 
# and reading_writing_data.py is in 'prioritizer/utils'
try:
    from prioritizer.utils.reading_writing_data import read_city_inventory, read_actions
    # Functions from prioritizer.py (now core.py) that we want to test or use for testing
    from prioritizer.prioritizer import (
        # initialize_openrouter_client_headers, # Removed - called internally now
        rank_actions_with_llm,         
        load_action_pairs_from_file, 
        ACTION_PAIRS_FILE_PATH,      
        get_openrouter_completion, # Added for direct testing
        DEFAULT_MODEL_NAME
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure that test_prioritizer.py is in the 'prioritizer' directory,")
    print("and that the project structure matches expectations (e.g., 'prioritizer/core.py', 'prioritizer/utils/reading_writing_data.py').")
    print(f"Current sys.path: {sys.path}")
    print(f"PROJECT_ROOT used for path: {PROJECT_ROOT}")
    sys.exit(1)


from dotenv import load_dotenv
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"OPENROUTER_API_KEY: {OPENROUTER_API_KEY}")
def run_single_pair_test():
    """
    Loads necessary data, selects one pair and one model, and runs the prioritization.
    """
    print("=========================================")
    print(" Running Prioritizer Test Script ")
    print("=========================================")

    # 1. Check API Key (Headers initialized internally by rank_actions_with_llm)
    print("\\n--- Checking for API Key ---")
    if not OPENROUTER_API_KEY:
        print("Test Error: OPENROUTER_API_KEY environment variable not set or loaded. Cannot proceed.")
        return
    print("OPENROUTER_API_KEY found.")


    # 2. Load all climate actions data (still useful for picking test pair data)
    print("\\n--- Loading All Climate Actions Data ---")
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
    print(f"Test will use the default model in rank_actions_with_llm ('{DEFAULT_MODEL_NAME}') unless overridden.")
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
        "reason": None,
        "error_detail": None,
        "llm_raw_response": None
    }

    try:
        # Call the refactored function directly
        # Prepare the input dictionary
        comparison_input = {
            "CityLocode": city_locode,
            "ActionA": action_a_id,
            "ActionB": action_b_id
        }
        
        # Mock get_openrouter_completion for this test to control its output
        # and avoid actual API calls during this unit/integration test run.
        expected_winner_id = action_a_id # Let's assume A wins for this mock
        expected_reason = "This is a mock reason for the test."
        mock_llm_json_response = {"actionid": expected_winner_id, "reason": expected_reason}

        # We need to patch where get_openrouter_completion is LOOKED UP from, 
        # which is within the 'prioritizer.prioritizer' module when rank_actions_with_llm calls it.
        with patch('prioritizer.prioritizer.get_openrouter_completion', return_value=mock_llm_json_response) as mock_get_completion:
            # Pass comparison_input positionally (or as keyword) and model_name only if overriding
            if model_override_for_test:
                print(f"  Overriding model for test: {model_override_for_test}")
                # rank_actions_with_llm now returns a dict {"actionid": ..., "reason": ...}
                llm_output_dict = rank_actions_with_llm(comparison_input, model_name=model_override_for_test)
            else:
                llm_output_dict = rank_actions_with_llm(comparison_input)
            
            winner_action_id = llm_output_dict.get("actionid")
            winner_reason = llm_output_dict.get("reason")

            print(f"  Test (Mocked) LLM Chose: {winner_action_id}, Reason: {winner_reason}")
            test_result_payload["winner"] = winner_action_id
            test_result_payload["reason"] = winner_reason
            # llm_raw_response would be the raw JSON string if we had it;
            # since we're mocking the parsed dict, we might not have the raw string here easily.
            # For this test, it's acceptable if llm_raw_response remains None or is the dict.
            test_result_payload["llm_raw_response"] = mock_llm_json_response # Store the mocked dict
     
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

def test_get_openrouter_completion_header_handling():
    """
    Tests the header handling logic of get_openrouter_completion.
    Uses mocking to avoid actual API calls.
    """
    print("\n======================================================")
    print(" Running get_openrouter_completion Header Logic Test ")
    print("======================================================\n")

    # Store original API key from test_prioritizer's scope, used for messages
    # The actual key used by get_openrouter_completion is prioritizer.prioritizer.OPENROUTER_API_KEY
    original_test_scope_api_key = OPENROUTER_API_KEY 

    model_name = "test/model"
    prompt_messages = [{"role": "user", "content": "test prompt"}]
    # This is the URL get_openrouter_completion will try to POST to
    expected_api_url = "https://openrouter.ai/api/v1/chat/completions" 

    # Mock successful response structure that get_openrouter_completion expects
    # The "content" should be a JSON STRING that get_openrouter_completion will parse.
    mock_action_id = "test_action_id_from_mock"
    mock_reason = "This is a test reason from mock."
    # Ensure quotes within the JSON string are correctly escaped for the f-string.
    mock_llm_content_json_string = f'{{"actionid": "{mock_action_id}", "reason": "{mock_reason}"}}'
    mock_api_response_structure = { # Renamed from mock_api_response_json to avoid confusion with the string above
        "choices": [{"message": {"content": mock_llm_content_json_string}}]
    }

    # --- Test Case 1: No headers provided ---
    print("--- Test Case 1: No headers provided (API key IS expected to be set in prioritizer.py) ---")
    if not original_test_scope_api_key: # Check if key was loaded in *this* test file's env
        print("  INFO: OPENROUTER_API_KEY not found in test_prioritizer.py's environment. " +
              "The test relies on prioritizer.prioritizer.OPENROUTER_API_KEY being set.")
    
    with patch('requests.post') as mock_post_case1:
        mock_post_case1.return_value.status_code = 200
        mock_post_case1.return_value.json.return_value = mock_api_response_structure # Use the structure
        try:
            # Call with no headers argument
            result = get_openrouter_completion(model_name, prompt_messages) 
            mock_post_case1.assert_called_once()
            args, kwargs = mock_post_case1.call_args
            assert args[0] == expected_api_url, f"Expected URL {expected_api_url}, got {args[0]}"
            assert "Authorization" in kwargs["headers"], "Authorization header missing"
            assert "Bearer " in kwargs["headers"]["Authorization"], "Authorization doesn't seem to be a Bearer token"
            assert kwargs["headers"]["Content-Type"] == "application/json", "Default Content-Type incorrect"
            assert isinstance(result, dict), f"Result should be a dictionary, got {type(result)}"
            assert result.get("actionid") == mock_action_id, f"Expected actionid '{mock_action_id}', got '{result.get('actionid')}'"
            assert result.get("reason") == mock_reason, f"Expected reason '{mock_reason}', got '{result.get('reason')}'"
            print("  SUCCESS: requests.post called with correct default headers and result parsed to dict.")
        except Exception as e:
            print(f"  FAILURE: {e}")
            # If OPENROUTER_API_KEY in prioritizer.py is None, this might be a ValueError.
            # That specific case is tested in Test Case 4.


    # --- Test Case 2: Content-Type missing, Authorization provided ---
    print("\n--- Test Case 2: Content-Type missing, Authorization provided ---")
    with patch('requests.post') as mock_post_case2:
        mock_post_case2.return_value.status_code = 200
        mock_post_case2.return_value.json.return_value = mock_api_response_structure # Use the structure
        custom_auth = "Bearer customkey123"
        try:
            result = get_openrouter_completion(model_name, prompt_messages, headers={"Authorization": custom_auth})
            mock_post_case2.assert_called_once()
            args, kwargs = mock_post_case2.call_args
            assert args[0] == expected_api_url
            assert kwargs["headers"]["Authorization"] == custom_auth, "Provided Authorization header not used"
            assert kwargs["headers"]["Content-Type"] == "application/json", "Default Content-Type not applied"
            assert isinstance(result, dict), f"Result should be a dictionary, got {type(result)}"
            assert result.get("actionid") == mock_action_id, f"Expected actionid '{mock_action_id}', got '{result.get('actionid')}'"
            assert result.get("reason") == mock_reason, f"Expected reason '{mock_reason}', got '{result.get('reason')}'"
            print("  SUCCESS: requests.post called with provided Auth and default Content-Type, result parsed to dict.")
        except Exception as e:
            print(f"  FAILURE: {e}")

    # --- Test Case 3: Authorization missing, Content-Type provided (API key IS expected to be set) ---
    print("\n--- Test Case 3: Authorization missing, Content-Type provided (API key expected in prioritizer.py) ---")
    if not original_test_scope_api_key:
        print("  INFO: OPENROUTER_API_KEY not found in test_prioritizer.py's environment. " +
              "Test relies on prioritizer.prioritizer.OPENROUTER_API_KEY.")

    with patch('requests.post') as mock_post_case3:
        mock_post_case3.return_value.status_code = 200
        mock_post_case3.return_value.json.return_value = mock_api_response_structure # Use the structure
        custom_content_type = "application/vnd.custom+json"
        try:
            result = get_openrouter_completion(model_name, prompt_messages, headers={"Content-Type": custom_content_type})
            mock_post_case3.assert_called_once()
            args, kwargs = mock_post_case3.call_args
            assert args[0] == expected_api_url
            assert "Authorization" in kwargs["headers"], "Authorization header missing (should be from API key)"
            assert "Bearer " in kwargs["headers"]["Authorization"], "Default Authorization doesn't seem to be a Bearer token"
            assert kwargs["headers"]["Content-Type"] == custom_content_type, "Provided Content-Type not used"
            assert isinstance(result, dict), f"Result should be a dictionary, got {type(result)}"
            assert result.get("actionid") == mock_action_id, f"Expected actionid '{mock_action_id}', got '{result.get('actionid')}'"
            assert result.get("reason") == mock_reason, f"Expected reason '{mock_reason}', got '{result.get('reason')}'"
            print("  SUCCESS: requests.post called with default Auth (from API Key) and provided Content-Type, result parsed to dict.")
        except Exception as e:
            print(f"  FAILURE: {e}")


    # --- Test Case 4: Authorization missing AND API key in prioritizer.py is NOT set (expect ValueError) ---
    print("\n--- Test Case 4: Auth missing, API key in prioritizer.py forced to None (expect ValueError from get_openrouter_completion) ---")
    # We patch 'prioritizer.prioritizer.OPENROUTER_API_KEY' which is used by the imported get_openrouter_completion
    with patch('prioritizer.prioritizer.OPENROUTER_API_KEY', None): 
        with patch('requests.post') as mock_post_case4: # Mock post, though it shouldn't be called
            mock_post_case4.return_value.status_code = 200 
            mock_post_case4.return_value.json.return_value = mock_api_response_structure # Use the structure, though shouldn't be reached for value error
            try:
                get_openrouter_completion(model_name, prompt_messages, headers={"Content-Type": "application/json"})
                print("  FAILURE: ValueError was NOT raised when Auth header and API Key were missing.")
            except ValueError as e:
                expected_error_msg = "OPENROUTER_API_KEY environment variable not set and 'Authorization' header was not provided."
                if expected_error_msg in str(e):
                    print(f"  SUCCESS: Correct ValueError raised: '{e}'")
                else:
                    print(f"  FAILURE: Incorrect ValueError message. Expected to contain '{expected_error_msg}', got: '{e}'")
            except Exception as e:
                print(f"  FAILURE: Unexpected exception type raised: {type(e).__name__} - {e}")
    
    print("\n======================================================")
    print(" get_openrouter_completion Header Logic Test Finished ")
    print("======================================================\n")

def test_rank_actions_with_non_strict_model_response():
    """
    Tests rank_actions_with_llm's ability to handle responses from models 
    that might not strictly adhere to returning ONLY the ActionID.
    It specifically checks the validation step in rank_actions_with_llm.
    """
    print("\n==================================================================")
    print(" Running Test: rank_actions_with_llm with Non-Strict Model Response ")
    print("==================================================================\n")

    test_model = "microsoft/phi-3-mini-128k-instruct:free" 
    action_a_id = "test_action_A_phi" 
    action_b_id = "test_action_B_phi"
    city_locode = "XX XXX"

    comparison_input = {
        "CityLocode": city_locode,
        "ActionA": action_a_id,
        "ActionB": action_b_id
    }

    # Mock data that rank_actions_with_llm would normally fetch
    mock_city_data = {"locode": city_locode, "name": "Test City for Phi"}
    mock_action_a_data = {"ActionID": action_a_id, "ActionName": "Test Action A for Phi"}
    mock_action_b_data = {"ActionID": action_b_id, "ActionName": "Test Action B for Phi"}
    mock_all_actions = {action_a_id: mock_action_a_data, action_b_id: mock_action_b_data}

    # --- Scenario A: Model returns ONLY the valid ActionID (as requested by prompt) ---
    print(f"--- Scenario A: Model ({test_model}) output is a valid ActionID ---")
    # We patch get_openrouter_completion which is called by rank_actions_with_llm
    # It should return a DICTIONARY now (as if JSON parsing was successful)
    mock_reason_A = "Mock reason for Action A being chosen."
    mock_successful_parsed_json_A = {"actionid": action_a_id, "reason": mock_reason_A}
    with patch('prioritizer.prioritizer.get_openrouter_completion', return_value=mock_successful_parsed_json_A) as mock_llm_call_A, \
         patch('prioritizer.prioritizer.read_city_inventory', return_value=mock_city_data) as mock_read_city, \
         patch('prioritizer.prioritizer.load_all_actions_data', return_value=mock_all_actions) as mock_load_actions, \
         patch('prioritizer.prioritizer.initialize_openrouter_client_headers', return_value={"Authorization": "Bearer test"}):
        try:
            result_dict = rank_actions_with_llm(comparison_input, model_name=test_model)
            assert isinstance(result_dict, dict), "Result should be a dictionary."
            assert result_dict.get("actionid") == action_a_id, f"Expected actionid '{action_a_id}', got '{result_dict.get('actionid')}'"
            assert result_dict.get("reason") == mock_reason_A, f"Expected reason '{mock_reason_A}', got '{result_dict.get('reason')}'"
            mock_llm_call_A.assert_called_once()
            print(f"  SUCCESS: rank_actions_with_llm returned correct dict for valid parsed JSON input.")
        except Exception as e:
            print(f"  FAILURE: Unexpected exception: {type(e).__name__} - {e}")

    # --- Scenario B: Model provides valid JSON, but 'actionid' is not one of the pair ---
    print(f"\n--- Scenario B: Model ({test_model}) provides JSON with an invalid 'actionid' ---")
    invalid_action_id = "completely_invalid_id"
    mock_reason_B = "Mock reason for an invalid action ID."
    mock_parsed_json_with_invalid_id = {"actionid": invalid_action_id, "reason": mock_reason_B}
    with patch('prioritizer.prioritizer.get_openrouter_completion', return_value=mock_parsed_json_with_invalid_id) as mock_llm_call_B, \
         patch('prioritizer.prioritizer.read_city_inventory', return_value=mock_city_data) as mock_read_city2, \
         patch('prioritizer.prioritizer.load_all_actions_data', return_value=mock_all_actions) as mock_load_actions2, \
         patch('prioritizer.prioritizer.initialize_openrouter_client_headers', return_value={"Authorization": "Bearer test"}):
        try:
            rank_actions_with_llm(comparison_input, model_name=test_model)
            print(f"  FAILURE: rank_actions_with_llm did NOT raise ValueError for JSON with invalid actionid: '{invalid_action_id}'")
        except ValueError as e:
            expected_error_fragment = f"LLM returned an invalid ActionID '{invalid_action_id}' in JSON"
            if expected_error_fragment in str(e):
                print(f"  SUCCESS: rank_actions_with_llm correctly raised ValueError: {e}")
            else:
                print(f"  FAILURE: ValueError message incorrect. Expected to contain '{expected_error_fragment}', got '{e}'")
        except Exception as e:
            print(f"  FAILURE: Unexpected exception type: {type(e).__name__} - {e}")

    # --- Scenario C: get_openrouter_completion fails (e.g., malformed JSON from LLM) ---
    print(f"\n--- Scenario C: get_openrouter_completion itself raises ValueError (e.g., malformed JSON string from LLM) ---")
    error_message_from_get_completion = "LLM response was not valid JSON. Raw response: '{{not_json'"
    with patch('prioritizer.prioritizer.get_openrouter_completion', side_effect=ValueError(error_message_from_get_completion)) as mock_llm_call_C, \
         patch('prioritizer.prioritizer.read_city_inventory', return_value=mock_city_data) as mock_read_city3, \
         patch('prioritizer.prioritizer.load_all_actions_data', return_value=mock_all_actions) as mock_load_actions3, \
         patch('prioritizer.prioritizer.initialize_openrouter_client_headers', return_value={"Authorization": "Bearer test"}):
        try:
            rank_actions_with_llm(comparison_input, model_name=test_model)
            print(f"  FAILURE: rank_actions_with_llm did NOT propagate ValueError from get_openrouter_completion.")
        except ValueError as e:
            if error_message_from_get_completion in str(e):
                print(f"  SUCCESS: rank_actions_with_llm correctly propagated ValueError from get_openrouter_completion: {e}")
            else:
                print(f"  FAILURE: Propagated ValueError message incorrect. Expected '{error_message_from_get_completion}', got '{e}'")
        except Exception as e:
            print(f"  FAILURE: Unexpected exception type when get_openrouter_completion failed: {type(e).__name__} - {e}")

    print("\n==================================================================")
    print(" Finished Test: rank_actions_with_llm with Non-Strict Model Response ")
    print("==================================================================\n")

if __name__ == "__main__":
    # Need to ensure DEFAULT_MODEL_NAME is available in this scope if it's used directly
    # However, test_prioritizer.py already imports it from prioritizer.prioritizer effectively
    # by way of other imports, or it can be imported directly if needed for print statements.
    # For the print statement in run_single_pair_test, it refers to DEFAULT_MODEL_NAME used *within* rank_actions_with_llm.
    # To make it explicit and available for the print: 
    from prioritizer.prioritizer import DEFAULT_MODEL_NAME 

    run_single_pair_test() 
    test_get_openrouter_completion_header_handling()
    test_rank_actions_with_non_strict_model_response() # Call the new test function
 