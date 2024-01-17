import os
import json
import sys
current_directory = os.getcwd()
print(current_directory)
sys.path.append(current_directory)
from openai import OpenAI
from math import exp
import numpy as np
from utility.env_manager import get_env_manager
from evaluation._data_generation import get_completion
from evaluation._data_generation import file_reader
import random

env_manager = get_env_manager()
client = OpenAI(api_key=env_manager['openai_keys']['OPENAI_API_KEY'])


def evaluate(prompt: str, user_message: str, context: str, use_test_data: bool = False) -> str:
    """Return the classification of the hallucination.
    @parameter prompt: the prompt to be completed.
    @parameter user_message: the user message to be classified.
    @parameter context: the context of the user message.
    @returns classification: the classification of the hallucination.
    """
    num_test_output = str(10)
    API_RESPONSE = get_completion(
        [
            {
                "role": "system", 
                "content": prompt.replace("{Context}", context).replace("{Question}", user_message)
            }
        ],
        model=env_manager['vectordb_keys']['VECTORDB_MODEL'],
        logprobs=True,
        top_logprobs=1,
    )

    system_msg = str(API_RESPONSE.choices[0].message.content)

    for i, logprob in enumerate(API_RESPONSE.choices[0].logprobs.content[0].top_logprobs, start=1):
        output = f'\nhas_sufficient_context_for_answer: {system_msg}, \nlogprobs: {logprob.logprob}, \naccuracy: {np.round(np.exp(logprob.logprob)*100,2)}%\n'
        print(output)
        if system_msg == 'true' and np.round(np.exp(logprob.logprob)*100,2) >= 95.00:
            classification = 'true'
        elif system_msg == 'false' and np.round(np.exp(logprob.logprob)*100,2) >= 95.00:
            classification = 'false'
        else:
            classification = 'false'
    return classification



def monte_carlo_eval(prompt):
    # Simulating different types of responses
    response_types = ['highly relevant', 'somewhat relevant', 'irrelevant']
    scores = {'highly relevant': 3, 'somewhat relevant': 2, 'irrelevant': 1}

    # Perform multiple random trials
    trials = 100
    total_score = 0
    for _ in range(trials):
        response = random.choice(response_types)
        total_score += scores[response]

    # Average score represents the evaluation
    return total_score / trials

def elo_eval(prompt, base_rating=1500):
    # Simulate the outcome of the prompt against standard criteria
    # Here, we randomly decide if the prompt 'wins', 'loses', or 'draws'
    outcomes = ['win', 'loss', 'draw']
    outcome = random.choice(outcomes)

    # Elo rating formula parameters
    K = 30  # Maximum change in rating
    R_base = 10 ** (base_rating / 400)
    R_opponent = 10 ** (1600 / 400)  # Assuming a fixed opponent rating
    expected_score = R_base / (R_base + R_opponent)

    # Calculate the new rating based on the outcome
    actual_score = {'win': 1, 'loss': 0, 'draw': 0.5}[outcome]
    new_rating = base_rating + K * (actual_score - expected_score)

    return new_rating

def evaluate_prompt(main_prompt, test_cases):
    evaluations = {}

    # Evaluate the main prompt using Monte Carlo and Elo methods
    evaluations['main_prompt'] = {
        'Monte Carlo Evaluation': monte_carlo_eval(main_prompt),
        'Elo Rating Evaluation': elo_eval(main_prompt)
    }

    # Evaluate each test case
    for idx, test_case in enumerate(test_cases):
        evaluations[f'test_case_{idx+1}'] = {
            'Monte Carlo Evaluation': monte_carlo_eval(test_case),
            'Elo Rating Evaluation': elo_eval(test_case)
        }

    return evaluations


if __name__ == "__main__":
    context_message = file_reader("prompts/context.txt")
    prompt_message = file_reader("prompts/generic-evaluation-prompt.txt")
    context = str(context_message)
    prompt = str(prompt_message)
    
    # user_message = str(input("question: "))

    
    # print(evaluate(prompt, user_message, context))

    # Example usage
    main_prompt = "why we use OepenAI?"
    test_cases = ["Who founded OpenAI?", 
                  "What was the initial goal of OpenAI?",
                    "What did OpenAI release in 2016?", 
                   "What project did OpenAI showcase in 2018?",
                   "How did the AI agents in OpenAI Five work together?"
                   ]
    result = evaluate_prompt(main_prompt, test_cases)
    print(result)

    {'main_prompt': {
        'Monte Carlo Evaluation': 1.94, 
        'Elo Rating Evaluation': 1489.2019499940866}, 
        'test_case_1': {'Monte Carlo Evaluation': 2.06, 'Elo Rating Evaluation': 1519.2019499940866}, 
        'test_case_2': {'Monte Carlo Evaluation': 2.02, 'Elo Rating Evaluation': 1519.2019499940866}, 
        'test_case_3': {'Monte Carlo Evaluation': 1.89, 'Elo Rating Evaluation': 1504.2019499940866}, 
        'test_case_4': {'Monte Carlo Evaluation': 1.98, 'Elo Rating Evaluation': 1519.2019499940866}, 
        'test_case_5': {'Monte Carlo Evaluation': 2.03, 'Elo Rating Evaluation': 1519.2019499940866}}