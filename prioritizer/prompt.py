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
            You are a climate action expert, tasked to prioritize and recommend the top action for a city  in following format: best_action_id: <action_id> based on the following guidelines:
            
            ### Guidelines for Action Prioritization the order is in the order of importance:
            1. **Emissions Reduction:** Actions that achieve significant greenhouse gas (GHG) emissions reduction should rank higher, especially those targeting the city's largest emission sectors. Those are the most important factor in the scoring. Action that gets a lot of emission of the city most emission intensive sector should always be scored better
            2. **Sector Relevance:** Actions targeting high-emission or priority sectors for the city should rank higher.

            3. **Environmental Compatibility:** Actions that align with the city's environment, such as biome and climate, should be preferred.
            4. **Cost-effectiveness:** Actions with lower costs and high benefits should rank higher.
            5. **Risk Reduction:** Prioritize actions that address climate hazards and reduce risks for the city effectively.
            6. **Socio-Demographic Suitability:** Actions should match the population size, density, and socio-economic context of the city.
            7. **Implementation Timeline:** Actions with shorter implementation timelines or faster impact should rank higher. Take into account the cost of the action and the time it takes to implement it at once.
            8. **Dependencies:** Actions with fewer dependencies or preconditions should be prioritized. 
            9. **City Size and Capacity:** Actions should be suitable for the city's capacity and resources to implement.

            ### Instructions:
            - Based on the rules, evaluate the two actions
            - Consider both qualitative and quantitative aspects of the actions.
            - Remember that data is provided in the order of importance and you should consider emissions and sectors first the 
            - Provide a detailed explanation for why action won
            ###Explanation:
            - The explanation should not mention the quantitative score, but rather the qualitative reasoning.
            - The explanation should mention both good and bad aspects of the action depending on which position it has in the ranking.
            - The explanation should be concise and to the point
            - Avoid decisive statements like "This action is the best" or "This action is the worst".
            - you have to avoid mentoning any numbers in the explanation.
            - The explanation is for the end costumer that is not a climate expert.
            - The explanation should enable user to understand the action strong and week parts for this particular city and why it was prioritized.
            - Focus on the city's context and how the action fits into that.
            - Return only the one best action in following format best_action_id: <action_id>
            
            ### Action Data (Top 20 Actions):
            {action}

            ### City Data:
            {city}

            RETURN ACTION ID OF THE BEST ACTION IN FORMAT: best_action_id: <action_id> SO I CAN FIND IT
            """

    return prompt
