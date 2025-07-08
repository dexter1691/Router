""" Contains functions for generating embeddings and returning the top k values of a dot product between a and b."""

import numpy as np
import ollama
from router.utils import timer
import json
from router.evals import Evals, recall_at_k
import matplotlib.pyplot as plt
import argparse
from sentence_transformers import SentenceTransformer

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

def generate_embeddings_sentence_transformer(model: SentenceTransformer, input_list: list[str]) -> np.ndarray:
    """ Generate embeddings for a list of strings using a sentence transformer model.
    Args:
        model_name: The name of the sentence transformer model to use.
        input_list: A list of strings to generate embeddings for.
    Returns:
        A numpy array of embeddings.
    """
    with timer("embeddings"):
        embeddings = model.encode(input_list)
        # normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.array(embeddings)

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


def run_evals(evals, predicted_integrations_list, top_k, score_threshold) -> tuple[float, float]:
    recall_at_k_integrations_list = []

    for q_idx, eval_instance in enumerate(evals.get_data()):
        expected_integrations = eval_instance["integrations"]
        prediction_integrations = predicted_integrations_list[q_idx]
        if expected_integrations == []:
            continue
        r_integrations = recall_at_k(prediction_integrations, expected_integrations, top_k, score_threshold)
        recall_at_k_integrations_list.append(r_integrations)

    return np.mean(recall_at_k_integrations_list).item()


def plot_results(results_integrations, model_name, score_thresholds, top_k_values, save_fig=True):
    plt.figure(figsize=(10, 6))

    for i, threshold in enumerate(score_thresholds):
        plt.plot(top_k_values, results_integrations[i], marker='o', label=f'Score threshold = {threshold}')
    plt.ylim(0.6, 1)
    plt.xlabel('Top-k')
    plt.ylabel('Recall')
    plt.title('Integration Recall at Different Score Thresholds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(top_k_values)
    plt.title(f"Integration Recall at Different Score Thresholds {model_name}")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"data/figures/results_{model_name}.png")
    else:
        plt.show()

def plot_results_comparisons(results_integrations_1, results_integrations_2, model_name_1, model_name_2, score_threshold, top_k_values, save_fig=True):
    plt.figure(figsize=(10, 6))
    if results_integrations_1 is not None:
        plt.plot(top_k_values, results_integrations_1[0], marker='o', label=f'{model_name_1}', color='orange')
    if results_integrations_2 is not None:
        plt.plot(top_k_values, results_integrations_2[0], marker='o', label=f'{model_name_2}', color='blue')
    plt.ylim(0.6, 1)
    plt.xlabel('Top-k')
    plt.ylabel('Recall')
    plt.title('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(top_k_values)
    plt.title(f"{model_name_1} vs {model_name_2} at thresold={score_threshold}")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"data/figures/results_{model_name_1}_and_{model_name_2}.png")
    else:
        plt.show()

def generate_predictions(model_name: str, evals: Evals):
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

    return predicted_integrations_list

def generate_predictions_sentence_transformer(model_name: str, evals: Evals, tool_descriptions_path: str = "data/tool_descriptions.json"):
    tools_description_json = json.load(open(tool_descriptions_path))
    tools_descriptions = list(tools_description_json.values())
    integrations_name = list(tools_description_json.keys())

    print(f'Length of tools_descriptions: {len(tools_descriptions)}')
    print(f'Length of integrations_name: {len(integrations_name)}')
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    tools_embeddings = generate_embeddings_sentence_transformer(
        model=model,
        input_list=tools_descriptions
    )
    predicted_integrations_list = []
    for query in evals.get_queries():
        query_embeddings = generate_embeddings_sentence_transformer(model, [query])
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        top_k_indices, top_k_scores = return_top_k(tools_embeddings, query_embeddings, len(tools_descriptions))
        predicted_integrations = []
        
        for k in range(len(tools_descriptions) - 1, -1, -1):
            integration_idx = top_k_indices[0, k]
            integration_name = integrations_name[integration_idx]
            # print(f"#{k}: Integration: {integration_name} Score: {top_k_scores[0, k]}")
            predicted_integrations.append((integration_name, top_k_scores[0, k]))
        # print("-"*100)
        predicted_integrations_list.append(predicted_integrations)
    
    return predicted_integrations_list


def main(model_name: str, evals: Evals):
    predicted_integrations_list = generate_predictions(model_name, evals)
    
    results_tools = []
    results_integrations = []

    score_thresholds = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    top_k_values = [2, 5, 10, 20, 27]
    
    for score_threshold in score_thresholds:
        result_for_threshold_integrations = []
        for top_k in top_k_values:
            r_integrations = run_evals(evals, predicted_integrations_list, top_k=top_k, score_threshold=score_threshold)
            result_for_threshold_integrations.append(r_integrations)
        results_integrations.append(result_for_threshold_integrations)

    plot_results(results_integrations, model_name, score_thresholds, top_k_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mxbai-embed-large")
    parser.add_argument("--evals_path", type=str, default="data/evals_test_500_v0.json")
    args = parser.parse_args()
    evals = Evals(args.evals_path)
    main(args.model_name, evals)