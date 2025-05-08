import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

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



ACTION_PAIRS_FILE_PATH = Path(__file__).resolve().parent / "data/expert_comparisons/comparisons_Martha_Dillon_C40.json"
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

def get_openrouter_completion(model_name: str, prompt_messages: list, headers: dict | None = None) -> dict:
    """
    Makes a call to the OpenRouter /chat/completions endpoint.
    If headers are not provided or incomplete, defaults will be used for Authorization and Content-Type.
    The response content is expected to be a JSON string, which will be parsed into a dictionary.

    Args:
        model_name: The name of the model to use.
        prompt_messages: A list of message objects for the prompt.
        headers: Optional. The authorization and content-type headers for the API call.
                 If None or incomplete, defaults will be applied.

    Returns:
        A dictionary parsed from the LLM's JSON response, expected to contain 'actionid' and 'reason'.

    Raises:
        requests.exceptions.RequestException: If the API call fails.
        KeyError, IndexError: If the response format is unexpected (before JSON parsing).
        ValueError: If the response content is empty, JSON decoding fails,
                    OPENROUTER_API_KEY is not set and Authorization header is missing,
                    or if the parsed JSON does not contain 'actionid' and 'reason' keys.
    """
    final_headers = {}
    if headers is not None:
        final_headers.update(headers)

    # Ensure Authorization header
    if "Authorization" not in final_headers:
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set and 'Authorization' header was not provided."
            )
        final_headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"

    # Ensure Content-Type header for JSON payload, if not already set
    if "Content-Type" not in final_headers:
        final_headers["Content-Type"] = "application/json"

    payload = {
        "model": model_name,
        "messages": prompt_messages,
    }

    response = requests.post(
        f"{OPENROUTER_API_BASE_URL}/chat/completions",
        headers=final_headers,  # Use the processed final_headers
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
            
    try:
        # First attempt to parse directly
        parsed_json = json.loads(content)
    except json.JSONDecodeError as e1:
        # If direct parsing fails, try to strip markdown code block if present
        stripped_content = content.strip()
        if stripped_content.startswith("```json") and stripped_content.endswith("```"):
            # Extract content between ```json\n and ```
            # Common pattern is ```json\n{...}\n``` or just ```json\n{...}```
            # We'll find the first { and last }
            first_brace = stripped_content.find('{')
            last_brace = stripped_content.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_substring = stripped_content[first_brace : last_brace + 1]
                try:
                    print(f"  Attempting to parse extracted JSON substring: '{json_substring[:100]}...'") # Log snippet
                    parsed_json = json.loads(json_substring)
                except json.JSONDecodeError as e2:
                    error_msg = f"LLM response was not valid JSON even after attempting to strip markdown. Extracted part: '{json_substring[:100]}...' Original raw: '{content[:100]}...'"
                    print(f"  Error: {error_msg} - Inner JSONDecodeError: {e2}")
                    raise ValueError(error_msg) from e2 # Raise error from trying to parse substring
            else:
                # Could not find valid braces in the supposed markdown block
                error_msg = f"LLM response started with ```json but could not extract a valid JSON object. Raw response: '{content[:100]}...'"
                print(f"  Error: {error_msg}")
                raise ValueError(error_msg) from e1 # Raise original error
        else:
            # Not a markdown code block, or not one we can handle, so original error stands
            error_msg = f"LLM response was not valid JSON. Raw response: '{content[:100]}...'"
            print(f"  Error: {error_msg} - JSONDecodeError: {e1}")
            raise ValueError(error_msg) from e1 # Raise original error

    if not isinstance(parsed_json, dict):
        error_msg = f"LLM response, when parsed, was not a dictionary as expected. Type: {type(parsed_json)}. Parsed: '{parsed_json}'"
        print(f"  Error: {error_msg}")
        raise ValueError(error_msg)
        
    if "actionid" not in parsed_json or "reason" not in parsed_json:
        error_msg = f"LLM JSON response missing 'actionid' or 'reason' key. Parsed JSON: '{parsed_json}'"
        print(f"  Error: {error_msg}")
        raise ValueError(error_msg)
            
    return parsed_json


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
            city_locode_raw = pair.get("CityLocode")
            city_locode_formatted = city_locode_raw # Default to raw
            if isinstance(city_locode_raw, str) and len(city_locode_raw) > 2:
                # Insert space after the first two characters, e.g., "BRCMG" -> "BR CMG"
                city_locode_formatted = f"{city_locode_raw[:2]} {city_locode_raw[2:]}"
            
            # Adjust keys to match comparisons_Martha_Dillon_C40.json
            mapped_pair = {
                "actionA_id": pair.get("ActionA"), # Changed from "actionA"
                "actionB_id": pair.get("ActionB"), # Changed from "actionB"
                "city_locode": city_locode_formatted # Use the formatted locode
            }
            # Ensure all required keys are present after mapping
            # Note: if city_locode_formatted is None (e.g. if pair.get("CityLocode") was None),
            # this check will correctly identify it.
            if not all(mapped_pair.values()): # Checks if any value is None or empty
                 print(f"Warning: Skipping pair due to missing data after mapping: {pair} -> {mapped_pair} (raw locode: {city_locode_raw})")
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

Based on your analysis, you MUST return a JSON object with two keys:
1. "actionid": The ActionID of the action you determine to be better.
2. "reason": A concise explanation (1-2 sentences) justifying your choice, focusing on the city's context and the action's suitability. This explanation is for a non-expert user.

Example of the required JSON output format:
{{
  "actionid": "ACTION_ID_OF_WINNER",
  "reason": "This action was chosen because it directly addresses the city's primary climate risks (e.g., flooding) and aligns with its key emission reduction goals in the transport sector, offering a good balance of impact and feasibility given the city's resources."
}}

Do not include any other text, explanation, or formatting outside of this single JSON object.

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
Return ONLY the JSON object in the format specified above. For example, if Action A (ID: {get_val(action_a_data, 'ActionID')}) is better, your response should look like:
{{
  "actionid": "{get_val(action_a_data, 'ActionID')}",
  "reason": "Action A is preferred because..."
}}
"""
    return prompt.strip()


# --- New LLM Invocation Function ---
def rank_actions_with_llm(
    comparison_input: dict, 
    model_name: str = DEFAULT_MODEL_NAME # Use the global default model name
) -> dict:
    """
    Compares two actions for a given city using an LLM.
    Fetches city and action data internally.
    The LLM is expected to return a JSON object with 'actionid' and 'reason'.

    Args:
        comparison_input: Dictionary containing {"CityLocode": str, "ActionA": ActionID_str, "ActionB": ActionID_str}.
        model_name: The name of the OpenRouter model to use.

    Returns:
        A dictionary containing {"actionid": winning_ActionID_str, "reason": reason_text_str}.

    Raises:
        ValueError: If input data is missing/invalid, if city/action data cannot be fetched,
                    or if the LLM response (after parsing) has an invalid ActionID.
        requests.exceptions.RequestException, KeyError: From get_openrouter_completion.
    """
    city_locode = comparison_input.get("CityLocode")
    action_a_id = comparison_input.get("ActionA")
    action_b_id = comparison_input.get("ActionB")

    if not all([city_locode, action_a_id, action_b_id]):
        raise ValueError("Missing 'CityLocode', 'ActionA', or 'ActionB' in comparison_input.")

    openrouter_headers = initialize_openrouter_client_headers() # Initialize headers internally
    
    # Ensure locode is a string before passing (to satisfy linter)
    assert isinstance(city_locode, str), "CityLocode must be a string."

    # Fetch data internally
    try:
        city_dict = read_city_inventory(city_locode)
    except ValueError as e: # Raised by read_city_inventory if not found
        raise ValueError(f"Failed to read city inventory for {city_locode}: {e}") from e
    
    all_actions_data = load_all_actions_data() # This function is defined in prioritizer.py
    if not all_actions_data:
        raise ValueError("Failed to load all_actions_data.")

    action_a_dict = all_actions_data.get(action_a_id)
    action_b_dict = all_actions_data.get(action_b_id)

    if not action_a_dict:
        raise ValueError(f"ActionID '{action_a_id}' (Action A) not found in loaded actions.")
    if not action_b_dict:
        raise ValueError(f"ActionID '{action_b_id}' (Action B) not found in loaded actions.")

    print(f"  Comparing Action {action_a_id} vs {action_b_id} for City {city_locode} using model {model_name}")
    
    # 1. Construct the prompt
    prompt_text = construct_comparison_prompt(action_a_dict, action_b_dict, city_dict)
    prompt_messages = [{"role": "user", "content": prompt_text}]

    # 2. Call the LLM completion function - it now returns a dict
    llm_response_dict = get_openrouter_completion(
        model_name=model_name,
        prompt_messages=prompt_messages,
        headers=openrouter_headers
    )
    
    # 3. Extract actionid and reason, then validate the actionid
    winner_action_id = llm_response_dict.get("actionid") # Assuming get_openrouter_completion validated presence
    reason_text = llm_response_dict.get("reason") # Also present, will be returned

    if winner_action_id not in [action_a_id, action_b_id]:
        error_msg = f"LLM returned an invalid ActionID '{winner_action_id}' in JSON. Expected one of '{action_a_id}' or '{action_b_id}'."
        # Optionally include more context: f"Full LLM response dict: {llm_response_dict}"
        raise ValueError(error_msg) 

    return llm_response_dict # Return the whole dict containing validated actionid and reason


# --- Core Prioritization Logic ---
def prioritize_action_pairs(
    action_pairs_list: list
    # `all_actions_data` parameter removed as data is fetched inside rank_actions_with_llm
) -> list:
    """
    Processes a list of action pairs, gets LLM comparison for each, and returns results.
    """
    results = []
    
    for i, pair_info in enumerate(action_pairs_list):
        action_a_id = pair_info.get("actionA_id")
        action_b_id = pair_info.get("actionB_id")
        city_locode = pair_info.get("city_locode")

        print(f"Processing pair {i+1}/{len(action_pairs_list)}: Action A ({action_a_id}) vs Action B ({action_b_id}) for City ({city_locode})")

        current_result_payload = {
            "actionA_id": action_a_id,
            "actionB_id": action_b_id,
            "city_locode": city_locode,
            "winner": None, 
            "reason": None,
            "error_detail": None 
        }

        if not all([action_a_id, action_b_id, city_locode]):
            print(f"  Skipping pair due to missing IDs or locode in input pair_info: {pair_info}")
            current_result_payload["winner"] = "Error: Missing input data in pair_info"
            current_result_payload["error_detail"] = "One or more of actionA_id, actionB_id, city_locode is missing from the input pair_info."
            results.append(current_result_payload)
            continue

        # Data fetching for action_a_data and action_b_data is now inside rank_actions_with_llm
        # City data is also fetched inside rank_actions_with_llm

        try:
            comparison_input = {
                "CityLocode": city_locode,
                "ActionA": action_a_id,
                "ActionB": action_b_id
            }
            # Call the LLM comparison function
            # Default model is used unless we pass model_name explicitly
            llm_result_dict = rank_actions_with_llm(comparison_input)
            
            winner_action_id = llm_result_dict.get("actionid")
            reason_text = llm_result_dict.get("reason")

            print(f"  LLM chose: {winner_action_id} because: {reason_text}")
            current_result_payload["winner"] = winner_action_id
            current_result_payload["reason"] = reason_text
        
        except ValueError as e: 
            error_msg = f"Error processing pair: {e}"
            print(f"  Error: {error_msg}")
            # Error classification based on message content from rank_actions_with_llm or get_openrouter_completion
            if "LLM response was not valid JSON" in str(e):
                current_result_payload["winner"] = "Error: LLM JSON Decode"
            elif "LLM JSON response missing 'actionid' or 'reason' key" in str(e):
                current_result_payload["winner"] = "Error: LLM JSON Structure"
            elif "LLM returned an invalid ActionID" in str(e): # Checks for invalid ActionID from rank_actions_with_llm
                current_result_payload["winner"] = "Error: Invalid LLM ActionID in JSON"
            elif f"Failed to read city inventory for {city_locode}" in str(e) or \
               f"ActionID '{action_a_id}' (Action A) not found" in str(e) or \
               f"ActionID '{action_b_id}' (Action B) not found" in str(e) or \
               "Failed to load all_actions_data" in str(e):
                current_result_payload["winner"] = "Error: Data fetching issue"
            elif "Missing 'CityLocode', 'ActionA', or 'ActionB'" in str(e):
                 current_result_payload["winner"] = "Error: Invalid input to ranker"
            else: 
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
        except KeyError as e: 
             error_msg = f"Error processing LLM response or invalid response format: {e}"
             print(f"  Error: {error_msg}")
             current_result_payload["winner"] = "Error: LLM processing error"
             current_result_payload["error_detail"] = error_msg
        except Exception as e: 
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

    # OPENROUTER_API_KEY is loaded at the module level now by dotenv.
    # The check below is still useful to ensure it was actually set in the .env
    if not OPENROUTER_API_KEY:
        print("Critical Error: OPENROUTER_API_KEY environment variable is not set or not loaded from .env.")
        print("Please set this variable in your .env file before running the script. Exiting.")
        return

    # Headers are now initialized inside rank_actions_with_llm, so this call is not strictly needed here
    # try:
    #     openrouter_headers = initialize_openrouter_client_headers()
    #     print("OpenRouter client headers (checked for presence of API key).")
    # except ValueError as e:
    #     print(f"Error during initialization: {e}")
    #     return

    print("--- Step 1: Loading All Climate Actions Data ---")
    # This is primarily to check if action data is accessible.
    # rank_actions_with_llm will load it again if this instance is not passed down (which it isn't currently).
    all_actions_data_check = load_all_actions_data()
    if not all_actions_data_check:
        print("Critical Error: Failed to load action data initially, or no actions were found.")
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
    print(f"\n--- Step 3: Prioritizing Action Pairs using LLM ---")
    
    # Model selection is handled within rank_actions_with_llm, which has a default.
    # Headers are no longer passed to prioritize_action_pairs
    prioritized_results = prioritize_action_pairs(
        action_pairs_list=current_action_pairs
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
