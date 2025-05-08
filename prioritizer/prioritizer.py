import json
import os
from pathlib import Path
import requests

# Attempt to import from utils. If prioritizer.py is at the root of hackday_q1,
# and utils is a subdirectory, this might need adjustment depending on how Python path is set.
# Assuming utils is in sys.path or PYTHONPATH is configured.
# If utils.reading_writing_data cannot be found, you might need to adjust the import
# or ensure the script is run in a way that Python can find the utils module
# e.g. by adding the project root to PYTHONPATH.
try:
    from utils.reading_writing_data import read_actions, read_city_inventory, write_output
except ImportError:
    print("Error: Could not import from utils.reading_writing_data.")

    raise # Re-raise the import error to stop execution if it cannot be resolved


# --- Configuration ---
# It's good practice to use environment variables for API keys.
# Ensure OPENROUTER_API_KEY is set in your environment.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
# You can change this to any model available on OpenRouter
DEFAULT_MODEL_NAME = "google/gemini-flash-1.5" 



ACTION_PAIRS_FILE_PATH = Path(__file__).resolve().parent / "action_pairs.json"
MODEL_LIST_FILE_PATH = Path(__file__).resolve().parent / "model_list.json"


# --- OpenRouter API Functions ---
def initialize_openrouter_client_headers():
    """
    Initializes and returns headers for OpenRouter API calls.
    """
    if not OPENROUTER_API_KEY:
        # This error will be caught by the main function early.
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    return headers

def get_openrouter_completion(model_name: str, prompt_messages: list, headers: dict) -> str:
    """
    Makes a call to the OpenRouter /chat/completions endpoint.

    Args:
        model_name: The name of the model to use.
        prompt_messages: A list of message objects for the prompt.
        headers: The authorization headers for the API call.

    Returns:
        The content of the model's response, expected to be the winner ActionID.

    Raises:
        requests.exceptions.RequestException: If the API call fails.
        KeyError, IndexError: If the response format is unexpected.
        ValueError: If the response content is empty or JSON decoding fails.
    """
    payload = {
        "model": model_name,
        "messages": prompt_messages,
    }

    response = requests.post(
        f"{OPENROUTER_API_BASE_URL}/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=60  # Adding a timeout for the request
    )
    response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
    
    try:
        response_data = response.json()
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        print(f"Response text: {response.text}")
        raise ValueError(f"Failed to decode JSON response: {response.text}") from e
        
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"Unexpected response format from API. Full response: {response_data}")
        raise KeyError(f"Could not extract content from LLM response: {response_data}") from e
        
    if not content:
        print(f"Warning: Received empty content from LLM. Full response: {response_data}")
        raise ValueError(f"Received empty content from LLM. Response: {response_data}")
            
    return content.strip()


# --- Data Loading Functions ---
def load_all_actions_data() -> dict:
    """
    Loads all climate actions using the utility function and converts them 
    to a dictionary keyed by ActionID for efficient lookup.
    """
    print("Attempting to load actions using 'read_actions()'...")
    actions_list = read_actions() # This function is from utils.reading_writing_data
    actions_dict = {action["ActionID"]: action for action in actions_list if action.get("ActionID")}
    
    if not actions_dict:
        print("Warning: No actions loaded or actions are missing ActionID after processing.")
    else:
        print(f"Successfully loaded {len(actions_dict)} actions into a dictionary.")
    return actions_dict


# --- New Data Loading Functions for action_pairs.json and model_list.json ---
def load_action_pairs_from_file(file_path: Path) -> list:
    """
    Loads action pairs from a specified JSON file.
    Maps 'actionA' to 'actionA_id' and 'actionB' to 'actionB_id'.
    """
    if not file_path.exists():
        print(f"Warning: Action pairs file not found at {file_path}. Returning empty list.")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Map keys
        mapped_data = []
        for pair in data:
            mapped_pair = {
                "actionA_id": pair.get("actionA"),
                "actionB_id": pair.get("actionB"),
                "city_locode": pair.get("city_locode")
            }
            # Ensure all required keys are present after mapping
            if not all(mapped_pair.values()): # Checks if any value is None or empty
                 print(f"Warning: Skipping pair due to missing data after mapping: {pair} -> {mapped_pair}")
                 continue
            mapped_data.append(mapped_pair)
        
        print(f"Successfully loaded {len(mapped_data)} action pairs from {file_path}.")
        return mapped_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}. Returning empty list.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path}: {e}. Returning empty list.")
        return []

def load_models_from_file(file_path: Path) -> list:
    """
    Loads a list of model objects from a specified JSON file.
    Each object is expected to have a 'name' and 'has_structured_outputs'.
    """
    if not file_path.exists():
        print(f"Warning: Model list file not found at {file_path}. Returning empty list.")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            models = json.load(f)
        # Basic validation for expected structure (list of dicts with 'name')
        if not isinstance(models, list) or not all(isinstance(m, dict) and 'name' in m for m in models):
            print(f"Warning: Model list file {file_path} does not have the expected format (list of dicts with 'name'). Returning empty list.")
            return []
        print(f"Successfully loaded {len(models)} model(s) from {file_path}.")
        return models
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}. Returning empty list.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path}: {e}. Returning empty list.")
        return []


# --- Prompt Construction ---
def construct_comparison_prompt(action_a_data: dict, action_b_data: dict, city_context_data: dict) -> str:
    """
    Constructs a detailed prompt for the LLM to compare two actions for a given city.
    Asks the LLM to return ONLY the ActionID of the better action.
    """
    # Helper to safely extract and format data, defaulting to 'N/A'
    def get_val(data, key, default="N/A"):
        val = data.get(key, default)
        if isinstance(val, (dict, list)):
            return json.dumps(val, indent=2) # Pretty print for complex types
        return str(val) # Ensure string conversion

    prompt = f"""You are an expert in climate change adaptation and mitigation planning.
Your task is to compare two proposed climate actions for a specific city and determine which action is more suitable or impactful for that city.
Please analyze the provided city context and the details of Action A and Action B.
Based on your analysis, return ONLY the ActionID of the action you determine to be better.
Do not include any other text, explanation, reasoning, or formatting. Just the ActionID.

City Context:
- Name: {get_val(city_context_data, 'name')}
- Locode: {get_val(city_context_data, 'locode')}
- Population: {get_val(city_context_data, 'populationSize')}
- Country: {get_val(city_context_data, 'countryname')}
- Region: {get_val(city_context_data, 'regionName')}
- Biome: {get_val(city_context_data, 'biome')}
- Climate Risks Summary (e.g., from CCRA): {get_val(city_context_data, 'ccra')}
- Emissions Summary (e.g., totalEmissions, scope1Emissions): 
  - Total: {get_val(city_context_data, 'totalEmissions')}
  - Scope1: {get_val(city_context_data, 'scope1Emissions')}
  - Scope2: {get_val(city_context_data, 'scope2Emissions')}
(Additional city details might be available in the full city_context_data object passed if needed by the system but not fully itemized here for brevity in prompt example.)

Action A:
- ActionID: {get_val(action_a_data, 'ActionID')}
- ActionName: {get_val(action_a_data, 'ActionName')}
- Description: {get_val(action_a_data, 'Description')}
- Sector: {get_val(action_a_data, 'Sector')}
- Subsector: {get_val(action_a_data, 'Subsector')}
- Hazard(s) Addressed: {get_val(action_a_data, 'Hazard')}
- PrimaryPurpose: {get_val(action_a_data, 'PrimaryPurpose')}
- CoBenefits: {get_val(action_a_data, 'CoBenefits')}
- GHGReductionPotential: {get_val(action_a_data, 'GHGReductionPotential')}
- AdaptationEffectiveness: {get_val(action_a_data, 'AdaptationEffectiveness')}
- CostInvestmentNeeded: {get_val(action_a_data, 'CostInvestmentNeeded')}
- TimelineForImplementation: {get_val(action_a_data, 'TimelineForImplementation')}

Action B:
- ActionID: {get_val(action_b_data, 'ActionID')}
- ActionName: {get_val(action_b_data, 'ActionName')}
- Description: {get_val(action_b_data, 'Description')}
- Sector: {get_val(action_b_data, 'Sector')}
- Subsector: {get_val(action_b_data, 'Subsector')}
- Hazard(s) Addressed: {get_val(action_b_data, 'Hazard')}
- PrimaryPurpose: {get_val(action_b_data, 'PrimaryPurpose')}
- CoBenefits: {get_val(action_b_data, 'CoBenefits')}
- GHGReductionPotential: {get_val(action_b_data, 'GHGReductionPotential')}
- AdaptationEffectiveness: {get_val(action_b_data, 'AdaptationEffectiveness')}
- CostInvestmentNeeded: {get_val(action_b_data, 'CostInvestmentNeeded')}
- TimelineForImplementation: {get_val(action_b_data, 'TimelineForImplementation')}

Considering all the above information, which action is better for the specified city?
Return ONLY the ActionID of the better action. For example, if Action A is better, return: {get_val(action_a_data, 'ActionID')}
"""
    return prompt.strip()


# --- New LLM Invocation Function ---
def rank_actions_with_llm(
    city_dict: dict, 
    action_a_dict: dict, 
    action_b_dict: dict, 
    openrouter_headers: dict, 
    model_name: str = "google/gemini-2.5-pro-preview"
) -> str:
    """
    Compares two actions for a given city using an LLM.

    Args:
        city_dict: Dictionary containing the city context data.
        action_a_dict: Dictionary for Action A.
        action_b_dict: Dictionary for Action B.
        openrouter_headers: Headers for the OpenRouter API call.
        model_name: The name of the OpenRouter model to use.

    Returns:
        The ActionID of the winning action as determined by the LLM.

    Raises:
        Various exceptions from construct_comparison_prompt or get_openrouter_completion 
        (e.g., requests.exceptions.RequestException, ValueError, KeyError).
    """
    print(f"  Comparing Action {action_a_dict.get('ActionID')} vs {action_b_dict.get('ActionID')} for City {city_dict.get('locode')} using model {model_name}")
    
    # 1. Construct the prompt
    prompt_text = construct_comparison_prompt(action_a_dict, action_b_dict, city_dict)
    prompt_messages = [{"role": "user", "content": prompt_text}]

    # 2. Call the LLM completion function
    # Exceptions from get_openrouter_completion will propagate up
    winner_action_id = get_openrouter_completion(
        model_name=model_name,
        prompt_messages=prompt_messages,
        headers=openrouter_headers
    )
    
    # 3. Basic validation of the returned ID
    action_a_id = action_a_dict.get("ActionID")
    action_b_id = action_b_dict.get("ActionID")
    if winner_action_id not in [action_a_id, action_b_id]:
        error_msg = f"LLM returned an invalid/unexpected ActionID: '{winner_action_id}'. Expected one of '{action_a_id}' or '{action_b_id}'."
        # Raise a ValueError or a custom exception might be better, but for now use ValueError
        # This will be caught by the calling function (prioritize_action_pairs)
        raise ValueError(error_msg) 

    return winner_action_id


# --- Core Prioritization Logic ---
def prioritize_action_pairs(
    action_pairs_list: list, 
    all_actions_data: dict, 
    openrouter_headers: dict
) -> list:
    """
    Processes a list of action pairs, gets LLM comparison for each, and returns results.
    """
    results = []
    
    if not all_actions_data:
        print("Critical Error: 'all_actions_data' is empty or None. Cannot proceed with prioritization.")
        # Optionally, populate all results with errors or return immediately
        for pair_info in action_pairs_list:
             results.append({**pair_info, "winner": "Error: Prerequisite action data missing", "error_detail": "all_actions_data was empty"})
        return results

    for i, pair_info in enumerate(action_pairs_list):
        action_a_id = pair_info.get("actionA_id")
        action_b_id = pair_info.get("actionB_id")
        city_locode = pair_info.get("city_locode")

        print(f"Processing pair {i+1}/{len(action_pairs_list)}: Action A ({action_a_id}) vs Action B ({action_b_id}) for City ({city_locode})")

        current_result_payload = {
            "actionA_id": action_a_id,
            "actionB_id": action_b_id,
            "city_locode": city_locode,
            "winner": None, # Placeholder
            "error_detail": None # Placeholder
        }

        if not all([action_a_id, action_b_id, city_locode]):
            print(f"  Skipping pair due to missing IDs or locode: {pair_info}")
            current_result_payload["winner"] = "Error: Missing input data"
            current_result_payload["error_detail"] = "One or more of actionA_id, actionB_id, city_locode is missing."
            results.append(current_result_payload)
            continue

        action_a_data = all_actions_data.get(action_a_id)
        action_b_data = all_actions_data.get(action_b_id)
        
        if not action_a_data:
            print(f"  Error: ActionID '{action_a_id}' (Action A) not found in loaded actions.")
            current_result_payload["winner"] = f"Error: Action A ID not found"
            current_result_payload["error_detail"] = f"ActionID '{action_a_id}' not found in all_actions_data."
            results.append(current_result_payload)
            continue
        if not action_b_data:
            print(f"  Error: ActionID '{action_b_id}' (Action B) not found in loaded actions.")
            current_result_payload["winner"] = f"Error: Action B ID not found"
            current_result_payload["error_detail"] = f"ActionID '{action_b_id}' not found in all_actions_data."
            results.append(current_result_payload)
            continue

        try:
            # Call the LLM comparison function
            winner_action_id = rank_actions_with_llm(
                city_dict=read_city_inventory(city_locode), 
                action_a_dict=action_a_data, 
                action_b_dict=action_b_data,
                openrouter_headers=openrouter_headers
            )
            
            print(f"  LLM chose: {winner_action_id}")
            current_result_payload["winner"] = winner_action_id
        
        except ValueError as e: # Catch city reading errors or invalid LLM response from rank_actions
            error_msg = f"Error processing pair: {e}"
            print(f"  Error: {error_msg}")
            # Distinguish between city data error and LLM error if possible
            if f"City with locode '{city_locode}' not found" in str(e):
                current_result_payload["winner"] = "Error: City locode data issue"
            elif "LLM returned an invalid/unexpected ActionID" in str(e):
                current_result_payload["winner"] = "Error: Invalid LLM response"
            else: # Other ValueErrors
                current_result_payload["winner"] = "Error: Processing ValueError"
            current_result_payload["error_detail"] = error_msg
        except requests.exceptions.HTTPError as e:
            error_msg = f"API call failed with HTTPError: {e}. Response: {e.response.text if e.response else 'N/A'}"
            print(f"  Error: {error_msg}")
            current_result_payload["winner"] = "Error: API HTTPError"
            current_result_payload["error_detail"] = error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"API call failed with RequestException: {e}"
            print(f"  Error: {error_msg}")
            current_result_payload["winner"] = "Error: API RequestException"
            current_result_payload["error_detail"] = error_msg
        except KeyError as e: # Catches JSON key errors etc. from get_openrouter_completion
             error_msg = f"Error processing LLM response or invalid response format: {e}"
             print(f"  Error: {error_msg}")
             current_result_payload["winner"] = "Error: LLM processing error"
             current_result_payload["error_detail"] = error_msg
        except Exception as e: 
            # Catch any other unexpected error during city reading or LLM call
            error_msg = f"An unexpected error occurred processing pair: {e}"
            print(f"  Error: {error_msg}")
            current_result_payload["winner"] = "Error: Unexpected processing error"
            current_result_payload["error_detail"] = error_msg
        
        results.append(current_result_payload)
            
    return results


# --- Output/Saving Function ---
def save_results(results_data: list, filename: str):
    """
    Saves the prioritization results to a JSON file using the utility function.
    The `write_output` function from `utils.reading_writing_data` handles 
    directory creation and uses its internally defined `OUTPUT_PATH`.
    """
    if not results_data:
        print("No results to save.")
        return
    
    try:
        # write_output comes from utils.reading_writing_data
        # It saves to hackday_q1/data/prioritized/filename
        write_output(results_data, filename)
        # The write_output function already prints a success message.
    except Exception as e:
        # This is a fallback, as write_output has its own error handling.
        print(f"An unexpected error occurred while trying to save results with 'write_output': {e}")


# --- Main Execution Block ---
def main():
    """
    Main function to orchestrate the climate action prioritization process.
    """
    print("========================================================")
    print(" Climate Action Prioritization Script Initializing... ")
    print("========================================================")

    if not OPENROUTER_API_KEY:
        print("Critical Error: OPENROUTER_API_KEY environment variable is not set.")
        print("Please set this variable before running the script. Exiting.")
        return

    try:
        openrouter_headers = initialize_openrouter_client_headers()
        print("OpenRouter client headers initialized.")
    except ValueError as e: # Should be caught by the check above, but good for safety
        print(f"Error during initialization: {e}")
        return

    print("--- Step 1: Loading All Climate Actions Data ---")
    all_actions_data = load_all_actions_data()
    if not all_actions_data:
        print("Critical Error: Failed to load action data, or no actions were found.")
        print("Please check the 'data/climate_actions/merged.json' file and 'utils/reading_writing_data.py' paths.")
        print("Exiting.")
        return
    # print(f"Successfully loaded {len(all_actions_data)} actions.") # Already printed in load_all_actions_data

    # --- Step 2: Defining/Loading Action Pairs for Comparison ---
    print("\n--- Step 2: Defining/Loading Action Pairs for Comparison ---")
    current_action_pairs = load_action_pairs_from_file(ACTION_PAIRS_FILE_PATH)
    if not current_action_pairs:
        print(f"No action pairs loaded from {ACTION_PAIRS_FILE_PATH}. Check the file or path.")
        print("No action pairs loaded, cannot proceed. Exiting.")
        return

    # --- Step 3: Prioritizing Action Pairs using LLM ---
    print(f"\n--- Step 3: Prioritizing Action Pairs using LLM (default in rank_actions_with_llm) ---")
    
    # Model selection is now handled within rank_actions_with_llm or its callers if overriding default
    prioritized_results = prioritize_action_pairs(
        action_pairs_list=current_action_pairs,
        all_actions_data=all_actions_data,
        openrouter_headers=openrouter_headers
    )

    # --- Step 4: Saving Results ---
    print("--- Step 4: Saving Results ---")
    if prioritized_results:
        output_filename = "prioritized_action_results.json"
        # The save_results function will print success/failure.
        save_results(prioritized_results, output_filename)
    else:
        print("No results were generated from the prioritization process to save.")

    print("========================================================")
    print(" Climate Action Prioritization Script Finished. ")
    print("========================================================")

if __name__ == "__main__":
    # This structure allows the script to be run directly.
    # Ensure that your Python environment can find the 'utils' module.
    # If run from the root of 'hackday_q1', and 'utils' is a subdirectory,
    # you might execute with: python prioritizer.py (if cwd is root and root is in PYTHONPATH)
    # or: python -m prioritizer (if prioritizer is treated as part of a package structure, more complex setup)
    # Simplest is often to ensure 'hackday_q1' root is in PYTHONPATH or run from there.
    main()
