""" Contains functions for generating embeddings and returning the top k values of a dot product between a and b."""

import numpy as np
import ollama
from router.utils import timer
import json
from router.evals import evals, recall_at_k
import matplotlib.pyplot as plt
import argparse


TOOLS_JSON_PATH = "data/tools.json"

def generate_embeddings(model_name: str, input_list: list[str]) -> np.ndarray:
    """ Generate embeddings for a list of strings using an ollama model.
    Args:
        model_name: The name of the ollama model to use.
        input_list: A list of strings to generate embeddings for.
    Returns:
        A numpy array of embeddings.
    """
    with timer("embeddings"):
        embeddings = ollama.embed(
            model=model_name,
            input=input_list
        )
        return np.array(embeddings.embeddings)

def return_top_k(tools_embeddings: np.ndarray, query_embeddings : np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """ Return the top k values of a dot product between a and b.
    Args:
        tools_embeddings: A numpy array of embeddings (N, D) where N is the number of embeddings and D is the dimension of the embedding
        query_embeddings: A numpy array of embeddings (M, D) where M is the number of queries and D is the dimension of the embedding
        k: The number of top values to return.
    Returns:
        A list of the top k values and their scores.
    """

    dot_product_M_N = np.dot(query_embeddings, tools_embeddings.T)
    top_k_indices = np.argsort(dot_product_M_N, axis=1)[:, -k:] # (M, k)
    top_k_scores = dot_product_M_N[np.arange(dot_product_M_N.shape[0])[:, None], top_k_indices] # (M, k)
    return top_k_indices, top_k_scores


def run_evals(evals, predicted_tools_list, predicted_integrations_list, top_k, score_threshold) -> tuple[float, float]:
    recall_at_k_list = []
    recall_at_k_integrations_list = []

    for q_idx, eval_instance in enumerate(evals.get_data()):
        expected_integrations = eval_instance["integrations"]
        expected_tools = eval_instance["tools"]
        predictions = predicted_tools_list[q_idx]
        r = recall_at_k(predictions, expected_tools, top_k, score_threshold)
        
        prediction_integrations = predicted_integrations_list[q_idx]
        r_integrations = recall_at_k(prediction_integrations, expected_integrations, top_k, score_threshold)
        recall_at_k_list.append(r)
        recall_at_k_integrations_list.append(r_integrations)

    return np.mean(recall_at_k_list).item(), np.mean(recall_at_k_integrations_list).item()


def plot_results(results_integrations, model_name, score_thresholds, top_k_values):
    plt.figure(figsize=(10, 6))

    for i, threshold in enumerate(score_thresholds):
        plt.plot(top_k_values, results_integrations[i], marker='o', label=f'Score threshold = {threshold}')

    plt.xlabel('Top-k')
    plt.ylabel('Recall')
    plt.title('Integration Recall at Different Score Thresholds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(top_k_values)
    plt.title(f"Integration Recall at Different Score Thresholds {model_name}")
    plt.tight_layout()
    plt.savefig(f"data/figures/results_{model_name}.png")


def generate_predictions(model_name: str):
    """ Generate predictions for a list of queries using an ollama model.
    Args:
        model_name: The name of the ollama model to use.
        top_k: The number of top values to return.
    Returns:
        A list of the top k values and their scores.
    """

    with open(TOOLS_JSON_PATH, "r") as f:
        tools = json.load(f)

    tools_flat_list = []
    tools_map = []
    tools_to_integration_map = {}
    for category, tools in tools.items():
        for tool in tools:
            tools_flat_list.append(tool["description"])
            tools_map.append(tool["name"])
            tools_to_integration_map[tool["name"]] = category

    print('Length of tools_flat_list', len(tools_flat_list))

    tools_embeddings = generate_embeddings(
        model_name=model_name,
        input_list=tools_flat_list
    )

    queries = evals.get_queries()

    query_embeddings = generate_embeddings(
        model_name=model_name,
        input_list=queries
    )
    top_k_indices, top_k_scores = return_top_k(tools_embeddings, query_embeddings, len(tools_flat_list))

    predicted_integrations_list = []
    predicted_tools_list = []

    for q_idx, query in enumerate(queries):
        print(f"Query: {query}")
        predicted_integrations = []
        predicted_tools = []
        for k in range(len(tools_flat_list) - 1, -1, -1):
            tool_idx = top_k_indices[q_idx, k]
            tool = tools_flat_list[tool_idx]
            tool_name = tools_map[tool_idx]
            integration_name = tools_to_integration_map[tool_name]
            print(f"#{k}: Integration: {integration_name} Tool: {tool_name} Score: {top_k_scores[q_idx, k]}")
            predicted_integrations.append((integration_name, top_k_scores[q_idx, k]))
            predicted_tools.append((tool_name, top_k_scores[q_idx, k]))

        predicted_integrations_list.append(predicted_integrations,)
        predicted_tools_list.append(predicted_tools)

        print("-"*100)

    return predicted_tools_list, predicted_integrations_list


def main(model_name: str):
    predicted_tools_list, predicted_integrations_list = generate_predictions(model_name)
    
    results_tools = []
    results_integrations = []

    score_thresholds = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    top_k_values = [2, 5, 10, 20, 27]
    
    for score_threshold in score_thresholds:
        result_for_threshold_tools = []
        result_for_threshold_integrations = []
        for top_k in top_k_values:
            r_tools, r_integrations = run_evals(evals, predicted_tools_list, predicted_integrations_list, top_k=top_k, score_threshold=score_threshold)
            result_for_threshold_tools.append(r_tools)
            result_for_threshold_integrations.append(r_integrations)
        results_tools.append(result_for_threshold_tools)
        results_integrations.append(result_for_threshold_integrations)

    plot_results(results_integrations, model_name, score_thresholds, top_k_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mxbai-embed-large")
    args = parser.parse_args()
    main(args.model_name)