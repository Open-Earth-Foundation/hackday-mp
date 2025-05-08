def return_prompt(action, city):
    """
    Generate the prioritization prompt based on the action and city data.

    Args:
    action (DataFrame): DataFrame containing the top 20 actions to prioritize.
    city (DataFrame): DataFrame containing the city data.

    Returns:
    str: The formatted prioritization prompt.
    """
    # Generate the prioritization prompt
    prompt = f"""
             You are a climate action expert, tasked to prioritize and recommend the top action for a city based on the following guidelines.

            ### Guidelines for Action Prioritization (in order of importance):
            1.  **Emissions Reduction:** Actions that achieve significant greenhouse gas (GHG) emissions reduction should rank higher, especially those targeting the city's largest emission sectors. This is the most important factor. Actions that significantly reduce emissions from the city's most emission-intensive sector should always be scored highest on this criterion.
            2.  **Sector Relevance:** Actions targeting high-emission or priority sectors for the city should rank higher.
            3.  **Environmental Compatibility:** Actions that align with the city's environment, such as biome and climate, should be preferred.
            4.  **Cost-effectiveness:** Actions with lower costs and high benefits should rank higher.
            5.  **Risk Reduction:** Prioritize actions that address climate hazards and reduce risks for the city effectively.
            6.  **Socio-Demographic Suitability:** Actions should match the population size, density, and socio-economic context of the city.
            7.  **Implementation Timeline:** Actions with shorter implementation timelines or faster impact should rank higher, considering the cost of the action and the time it takes to implement it.
            8.  **Dependencies:** Actions with fewer dependencies or preconditions should be prioritized.
            9.  **City Size and Capacity:** Actions should be suitable for the city's capacity and resources to implement.

            ### Instructions for Evaluation:
            -   Based on the guidelines above, evaluate the provided actions to select the single best one.
            -   Consider both qualitative and quantitative aspects of the actions when applying the prioritization guidelines.
            -   Remember that the prioritization guidelines are ordered by importance; emissions reduction and sector relevance are the primary factors.

            ### Guidelines for the "reason" field in the output:
            -   The explanation should detail why the selected action was chosen as the best for the city.
            -   It should not mention any quantitative scores or numerical data, focusing instead on qualitative reasoning.
            -   Highlight both the positive aspects of the chosen action and any potential considerations or challenges related to it within the city's specific context.
            -   The explanation should be concise, clear, and to the point.
            -   Avoid decisive or overly strong statements like "This action is unequivocally the best."
            -   The language should be accessible to a non-expert end customer.
            -   Focus on the city's specific context (e.g., its primary emission sources, environmental characteristics, socio-demographic profile) and how the action aligns with or addresses these.
            -   The explanation should help the user understand the action's primary strengths and any weaknesses or trade-offs considered for this particular city and why it was ultimately prioritized.

            ### Action Data:
            {action}

            ### City Data:
            {city}

            ### Output Format:
            Your response MUST be a single JSON object in the following format:
            ```json
                {
                "reason": "<Your detailed qualitative explanation, following the guidelines above, for why this action was prioritized for the city.>",
                "actionid": "<The ID of the single best action selected>"
                }
            ```
            Ensure no other text or explanation precedes or follows this JSON object.
            """

    return prompt
