#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils_new.py
@Contact :   zongfang.liu@mbzuai.ac.ae
@License :   (C)Copyright 2022-2023, ZongfangLiu

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2024/9/13 16:08   zfliu      1.0         None
"""
from names_dataset import NameDataset
import random
import openai
import json
from pydantic import BaseModel
from prompt import *
import time
import os
import shutil
import heapq
from operator import itemgetter

import numpy as np

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.stats import pearsonr


client = openai.Client(api_key="xxx")


def getUniformDistribution(part_left, part_right):
    part_left = list(part_left)
    part_right = list(part_right)

    uniform_distribution = {}
    total_nodes = len(part_left) + len(part_right)

    for node in part_left + part_right:
        uniform_distribution[node] = 1 / total_nodes

    return uniform_distribution


def getNodesFromPartitionWithHighestDegree(G, k, part):
    return heapq.nlargest(k, G.degree(part), key=itemgetter(1))


def metric_random_walk_controversy_score(G, opinions, k=10, alpha=0.85, max_iter=200):
    positive_opinions = {node for node, opinion in opinions.items() if opinion > 0}
    negative_opinions = {node for node, opinion in opinions.items() if opinion <= 0}

    partition = {0: negative_opinions, 1: positive_opinions}
    part_left = partition[0]
    part_right = partition[1]

    uniform_left = getUniformDistribution(part_left, part_right)
    pagerank_left = nx.pagerank(G, alpha=alpha, personalization=uniform_left, dangling=uniform_left, max_iter=max_iter)

    uniform_right = getUniformDistribution(part_right, part_left)
    pagerank_right = nx.pagerank(G, alpha=alpha, personalization=uniform_right, dangling=uniform_right,
                                 max_iter=max_iter)


    top_nodes_left = getNodesFromPartitionWithHighestDegree(G, k, part_left)
    top_nodes_right = getNodesFromPartitionWithHighestDegree(G, k, part_right)

    start_left_end_left = sum([pagerank_left[k] for k, v in top_nodes_left])
    start_left_end_right = sum([pagerank_left[k] for k, v in top_nodes_right])
    start_right_end_left = sum([pagerank_right[k] for k, v in top_nodes_left])
    start_right_end_right = sum([pagerank_right[k] for k, v in top_nodes_right])

    left_ratio = float(len(part_left)) / G.number_of_nodes()
    right_ratio = float(len(part_right)) / G.number_of_nodes()

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0

    p_start_left_end_left = safe_divide(start_left_end_left * left_ratio,
                                        (start_left_end_left * left_ratio) + (start_right_end_left * right_ratio))
    p_start_left_end_right = safe_divide(start_left_end_right * left_ratio,
                                         (start_right_end_right * right_ratio) + (start_left_end_right * left_ratio))
    p_start_right_end_right = safe_divide(start_right_end_right * right_ratio,
                                          (start_right_end_right * right_ratio) + (start_left_end_right * left_ratio))
    p_start_right_end_left = safe_divide(start_right_end_left * right_ratio,
                                         (start_left_end_left * left_ratio) + (start_right_end_left * right_ratio))

    rwc_score = p_start_left_end_left * p_start_right_end_right - p_start_left_end_right * p_start_right_end_left


    return rwc_score


def metric_neighbors_correlation_index(G, opinions, mode='numeric'):
    """
    Calculate the Neighbors Correlation Index (NCI) for a graph G based on opinions.

    Parameters:
    G: networkx.Graph
        The input graph where nodes are individuals and edges represent connections.
    opinions: dict
        A dictionary of opinions where keys are node ids and values are the opinions (float).

    Returns:
    nci: float
        The Neighbors Correlation Index for the graph.
    """
    nodes = list(G.nodes())
    print(opinions)
    if mode == 'llm':
        node_opinions = np.array([opinions[f'{node}'] for node in nodes])

        neighbor_avg_opinions = []

        for node in nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 0:
                avg_opinion = np.mean([opinions[f'{neighbor}'] for neighbor in neighbors])
            else:
                avg_opinion = opinions[node]

            neighbor_avg_opinions.append(avg_opinion)

        neighbor_avg_opinions = np.array(neighbor_avg_opinions)

        nci, _ = pearsonr(node_opinions, neighbor_avg_opinions)

        return nci
    else:
        node_opinions = np.array([opinions[node] for node in nodes])

        neighbor_avg_opinions = []

        for node in nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 0:
                avg_opinion = np.mean([opinions[neighbor] for neighbor in neighbors])
            else:
                avg_opinion = opinions[node]

            neighbor_avg_opinions.append(avg_opinion)

        neighbor_avg_opinions = np.array(neighbor_avg_opinions)

        nci, _ = pearsonr(node_opinions, neighbor_avg_opinions)

        return nci

def metric_polarization(G, opinions):

    opinion_values = list(opinions.values())

    n = len(opinion_values)

    mean_opinion = np.mean(opinion_values)

    polarization = np.sum((np.array(opinion_values) - mean_opinion) ** 2) / len(opinion_values)

    return polarization


def metric_global_disagreement(G, opinions, mode = 'numeric'):

    global_disagreement = 0
    if mode == 'llm':

        for i in G.nodes():
            local_disagreement = 0
            for j in G.neighbors(i):
                weight_ij = G[i][j].get('weight', 1)

                opinion_diff = opinions[f'{i}'] - opinions[f'{j}']

                local_disagreement += weight_ij * (opinion_diff ** 2)

            global_disagreement += local_disagreement / len(list(G.neighbors(i)))

        global_disagreement *= 0.5
        global_disagreement /= len(opinions)

        return global_disagreement
    else:
        for i in G.nodes():
            local_disagreement = 0
            for j in G.neighbors(i):
                weight_ij = G[i][j].get('weight', 1)
                opinion_diff = opinions[i] - opinions[j]
                local_disagreement += weight_ij * (opinion_diff ** 2)

            global_disagreement += local_disagreement / len(list(G.neighbors(i)))

        global_disagreement *= 0.5
        global_disagreement /= len(opinions)

        return global_disagreement
class update_opinion_response(BaseModel):
    opinion: str
    belief: int
    reasoning: str

class reflecting_response(BaseModel):
    short_term_memory: str

class long_memory_response(BaseModel):
    long_term_memory: str


def probability_threshold(threshold):
    '''
    Used in self.infect_interaction()
    '''
    # Generates random number from 0 to 1

    return (np.random.rand() < threshold)


def generate_qualifications(n: int):
    '''
    Returns a list of random educational qualifications.

    Parameters:
    n (int): The number of qualifications to generate.
    '''

    # Define a list of possible qualifications including lower levels and no education
    qualifications = ['No Education', 'Primary School', 'Middle School',
                      'High School Diploma', 'Associate Degree', 'Bachelor\'s Degree',
                      'Master\'s Degree', 'PhD', 'Professional Certificate']

    # Randomly select n qualifications from the list
    generated_qualifications = random.choices(qualifications, k=n)

    return generated_qualifications


def generate_names(n: int, s: int, country_alpha2='US'):
    '''
    Returns random names as names for agents from top names in the USA
    Used in World.init to initialize agents
    '''

    # This function will randomly selct n names (n/2 male and n/2 female) without
    # replacement from the s most popular names in the country defined by country_alpha2
    if n % 2 == 1:
        n += 1
    if s % 2 == 1:
        s += 1

    nd = NameDataset()
    male_names = nd.get_top_names(s // 2, 'Male', country_alpha2)[country_alpha2]['M']
    female_names = nd.get_top_names(s // 2, 'Female', country_alpha2)[country_alpha2]['F']
    if s < n:
        raise ValueError(f"Cannot generate {n} unique names from a list of {s} names.")
    # generate names without repetition
    names = random.sample(male_names, k=n // 2) + random.sample(female_names, k=n // 2)
    del male_names
    del female_names
    random.shuffle(names)
    return names


def generate_big5_traits(n: int):
    '''
    Return big 5 traits for each agent
    Used in World.init to initialize agents
    '''

    # Trait generation
    agreeableness_pos = ['Cooperation', 'Amiability', 'Empathy', 'Leniency', 'Courtesy', 'Generosity', 'Flexibility',
                         'Modesty', 'Morality', 'Warmth', 'Earthiness', 'Naturalness']
    agreeableness_neg = ['Belligerence', 'Overcriticalness', 'Bossiness', 'Rudeness', 'Cruelty', 'Pomposity',
                         'Irritability',
                         'Conceit', 'Stubbornness', 'Distrust', 'Selfishness', 'Callousness']
    # Did not use Surliness, Cunning, Predjudice,Unfriendliness,Volatility, Stinginess

    conscientiousness_pos = ['Organization', 'Efficiency', 'Dependability', 'Precision', 'Persistence', 'Caution',
                             'Punctuality',
                             'Punctuality', 'Decisiveness', 'Dignity']
    # Did not use Predictability, Thrift, Conventionality, Logic
    conscientiousness_neg = ['Disorganization', 'Negligence', 'Inconsistency', 'Forgetfulness', 'Recklessness',
                             'Aimlessness',
                             'Sloth', 'Indecisiveness', 'Frivolity', 'Nonconformity']

    surgency_pos = ['Spirit', 'Gregariousness', 'Playfulness', 'Expressiveness', 'Spontaneity', 'Optimism', 'Candor']
    # Did not use Humor, Self-esteem, Courage, Animation, Assertion, Talkativeness, Energy level, Unrestraint
    surgency_neg = ['Pessimism', 'Lethargy', 'Passivity', 'Unaggressiveness', 'Inhibition', 'Reserve', 'Aloofness']
    # Did not use Shyness, Silenece

    emotional_stability_pos = ['Placidity', 'Independence']
    emotional_stability_neg = ['Insecurity', 'Emotionality']
    # Did not use Fear, Instability, Envy, Gullibility, Intrusiveness

    intellect_pos = ['Intellectuality', 'Depth', 'Insight', 'Intelligence']
    # Did not use Creativity, Curiousity, Sophistication
    intellect_neg = ['Shallowness', 'Unimaginativeness', 'Imperceptiveness', 'Stupidity']

    # Combine each trait
    agreeableness_tot = agreeableness_pos + agreeableness_neg
    conscientiousness_tot = conscientiousness_pos + conscientiousness_neg
    surgency_tot = surgency_pos + surgency_neg
    emotional_stability_tot = emotional_stability_pos + emotional_stability_neg
    intellect_tot = intellect_pos + intellect_neg

    # create traits list to be returned
    traits_list = []

    for _ in range(n):
        agreeableness_rand = random.choice(agreeableness_tot)
        conscientiousness_rand = random.choice(conscientiousness_tot)
        surgency_rand = random.choice(surgency_tot)
        emotional_stability_rand = random.choice(emotional_stability_tot)
        intellect_rand = random.choice(intellect_tot)

        selected_traits = [agreeableness_rand, conscientiousness_rand, surgency_rand,
                           emotional_stability_rand, intellect_rand]

        traits_chosen = (', '.join(selected_traits))
        traits_list.append(traits_chosen)
    del agreeableness_rand
    del conscientiousness_rand
    del surgency_rand
    del emotional_stability_rand
    del intellect_rand
    del selected_traits
    del traits_chosen
    return traits_list


def factorize(n):
    '''
    Factorize number for ideal grid dimensions for # of agents
    Used in World.init
    '''
    for i in range(int(n ** 0.5), 1, -1):
        if n % i == 0:
            return (i, n // i)
    return (n, 1)


def get_completion_from_messages_structured(messages, system_messages="You are a helpful assistant.",
                                            model="gpt-4o-2024-08-06", temperature=1, response_type=update_opinion_response):
    success = False
    retry = 0
    max_retries = 30

    while retry < max_retries and not success:
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_messages},
                    {"role": "user", "content": messages},
                ],
                temperature=temperature,
                response_format=response_type,
            )
            #print(completion)

            response_object = completion.choices[0].message.parsed
            success = True
            return response_object

        except Exception as e:
            print(f"Error: {e}\nRetrying...")
            retry += 1
            time.sleep(2)

    return None


def update_day(agent):

    user_msg = update_opinion_prompt.format(belief_keywords=agent.belief_keywords,
                                            belief=agent.belief,
                                            long_mem=agent.long_opinion_memory,
                                            topic=agent.topic,
                                            opinion=agent.opinions[-1])

    agent.opinion, agent.belief, agent.reasoning = agent.response_and_belief(user_msg, agent.gpt_model)
    agent.opinions.append(agent.opinion)
    agent.beliefs.append(agent.belief)
    agent.reasonings.append(agent.reasoning)


def extract_beliefs_at_all_steps(agent_data_file_path):
    with open(agent_data_file_path, "r") as file:
        agent_data = json.load(file)

    time_steps = len(next(iter(agent_data.values()))["beliefs"])
    beliefs_at_steps = {time_step: {} for time_step in range(time_steps)}

    for agent_id, agent_info in agent_data.items():
        for time_step in range(time_steps):
            beliefs_at_steps[time_step][agent_id] = agent_info["beliefs"][time_step]

    return beliefs_at_steps


def update_belief_plot(frame, G, pos, node_size, labels, ax, beliefs_at_steps, time_step_text):
    ax.clear()

    color_palette = sns.color_palette("coolwarm", n_colors=5)
    belief_colors = {
        -2: color_palette[0],
        -1: color_palette[1],
        0: color_palette[2],
        1: color_palette[3],
        2: color_palette[4],
    }

    beliefs = beliefs_at_steps[frame]
    colors = [belief_colors[beliefs[str(node)]] for node in G.nodes()]

    nx.draw(G, pos, node_size=node_size, labels=labels, node_color=colors, with_labels=True, edge_color='gray', font_size=10, ax=ax)
    time_step_text.set_text(f"Day {frame + 1}")


def generate_belief_animation(network_file_path, agents_interaction_data_file_path, output_file, network_type, show_label=True, fps=1):
    with open(network_file_path, "r") as file:
        network_data = json.load(file)

    G = nx.Graph()
    G.add_nodes_from(network_data["nodes"])
    G.add_edges_from(network_data["edges"])

    labels = {}
    if show_label:
        for node in G.nodes():
            labels[node] = str(node)
    else:
        for node in G.nodes():
            labels[node] = ""

    if network_type == 'small_world':
        graph_name = "Small world network structure."
        pos = nx.shell_layout(G)
    elif network_type == 'scale_free':
        graph_name = "Scale free network structure"
        pos = nx.kamada_kawai_layout(G)
    else:
        graph_name = "Random network structure"
        # pos = nx.spring_layout(G, seed=42)
        pos = nx.kamada_kawai_layout(G)

    node_size = [G.degree[n] * 80 for n in G.nodes()]

    beliefs_at_steps = extract_beliefs_at_all_steps(agents_interaction_data_file_path)
    total_steps = len(beliefs_at_steps)
    print(total_steps)

    fig, ax = plt.subplots(figsize=(8, 8))
    time_step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=14, verticalalignment='top')

    ani = FuncAnimation(fig, update_belief_plot, frames=total_steps,
                        fargs=(G, pos, node_size, labels, ax, beliefs_at_steps, time_step_text), interval=1000 / fps)

    ani.save(output_file, writer='pillow', fps=fps)


def visulize_opinions(network_file_path, agents_interaction_data_file_path, directory,
                      model_type, network_type, network_seed, show_label=True, step=0):

    with open(network_file_path, "r") as file:
        network_data = json.load(file)

    G = nx.Graph()
    G.add_nodes_from(network_data["nodes"])
    G.add_edges_from(network_data["edges"])

    beliefs_at_steps = extract_beliefs_at_all_steps(agents_interaction_data_file_path)

    opinions = beliefs_at_steps[step]
    file_prefix = f"llm_network_{network_type}_model_{model_type}_seed_{network_seed}_step_{step}.png"

    file_path = os.path.join(directory, file_prefix)

    labels = {}
    if show_label:
        for node in G.nodes():
            labels[node] = str(node)
    else:
        for node in G.nodes():
            labels[node] = ""

    if network_type == 'small_world':
        graph_name = "Small world network structure."
        pos = nx.shell_layout(G)
    elif network_type == 'scale_free':
        graph_name = "Scale free network structure"
        pos = nx.kamada_kawai_layout(G)
    else:
        graph_name = "Random network structure"
        # pos = nx.spring_layout(G, seed=42)
        pos = nx.kamada_kawai_layout(G)

    # Get node colors based on their opinions
    node_colors = [opinions[str(node)] for node in G.nodes()]

    # Get node sizes based on degree (number of edges)
    node_size = [G.degree[n] * 100 for n in G.nodes()]  # Scale degree to control size (adjust factor as needed)

    # Define color map and normalization range for opinions
    cmap = plt.get_cmap('coolwarm')  # You can experiment with other colormaps like 'plasma' or 'viridis'
    vmin = -2  # Minimum opinion value
    vmax = 2  # Maximum opinion value

    print(show_label)
    # Create the figure for visualization
    plt.figure(figsize=(10, 10))
    nx.draw(
        G, pos, labels=labels, with_labels=show_label, node_color=node_colors,
        cmap=cmap, node_size=node_size, edge_color='gray', vmin=vmin, vmax=vmax
    )

    # plt.title("Network Visualization with Leader Nodes Highlighted and Opinion-based Coloring")
    plt.title(f"{graph_name}")
    # Save or show the plot
    # if save_image:
    plt.savefig(file_path)
    plt.savefig(file_path.replace(".png", ".pdf"))
    # plt.show()
    plt.close()

def visulize_metrics(network_file_path, agents_interaction_data_file_path, directory,
                      model_type, network_type, network_seed, show_label=True, step=0):

    with open(network_file_path, "r") as file:
        network_data = json.load(file)

    G = nx.Graph()
    G.add_nodes_from(network_data["nodes"])
    G.add_edges_from(network_data["edges"])

    beliefs_at_steps = extract_beliefs_at_all_steps(agents_interaction_data_file_path)
    max_steps = step
    step = 0
    opinions = beliefs_at_steps[step]
    print(opinions)
    print(opinions)
    for node in G.nodes:
        print(node)

    scores_nci, scores_polarization, scores_gd , scores_rwc= [], [], [], []
    history_opinions = {node: [opinions[f'{node}']] for node in G.nodes()}
    score_nci = metric_neighbors_correlation_index(G, opinions)
    scores_nci.append(score_nci)

    score_polarization = metric_polarization(G, opinions)
    scores_polarization.append(score_polarization)

    score_gd = metric_global_disagreement(G, opinions)
    scores_gd.append(score_gd)
    print('here')

    for step in range(max_steps):
        new_opinions = beliefs_at_steps[step]

        for node in G.nodes():
            history_opinions[node].append(new_opinions[f'{node}'])

        score_nci = metric_neighbors_correlation_index(G, new_opinions)
        scores_nci.append(score_nci)

        score_polarization = metric_polarization(G, new_opinions)
        scores_polarization.append(score_polarization)

        score_gd = metric_global_disagreement(G, new_opinions)
        scores_gd.append(score_gd)

        opinions = new_opinions

    if step != max_steps - 1:
        step -= 1
    result_data = {
        'scores_nci': scores_nci,
        'scores_polarization': scores_polarization,
        'scores_gd': scores_gd,
        # 'scores_rwc': scores_rwc,
        'final_step': step
    }
    print(f"{network_type} result polarization: ", round(result_data['scores_polarization'][0],4), round(result_data['scores_polarization'][-1],4))
    print(f"{network_type} result gd: ", round(result_data['scores_gd'][0],4), round(result_data['scores_gd'][-1],4))
    print(f"{network_type} result: nci", round(result_data['scores_nci'][0],4), round(result_data['scores_nci'][-1],4))

    file_prefix = f"metric_llm_network_{network_type}_model_{model_type}_seed_{network_seed}_step_{step}_.png"

    file_path = os.path.join(directory, file_prefix)

    fig = plt.figure(figsize=(10, 6))

    color_nci = sns.color_palette("coolwarm", as_cmap=False)[4]
    color_initial = sns.color_palette("Spectral", as_cmap=False)[0]
    color_final = sns.color_palette("Spectral", as_cmap=False)[5]

    ax1 = fig.add_subplot(111, label="1")
    ax1.plot(range(len(result_data['scores_nci'])), result_data['scores_nci'], color=color_nci, marker="o", label="NCI", linewidth=2, markersize=8)
    ax1.set_xlabel("Steps (NCI)", fontsize=17, color=color_nci, labelpad=10)
    ax1.set_ylabel("NCI Score", fontsize=17, color=color_nci, labelpad=-3)
    ax1.tick_params(axis="x", colors=color_nci, labelsize=15)
    ax1.tick_params(axis="y", colors=color_nci, labelsize=15)
    ax1.tick_params(axis="x", colors=color_nci)
    ax1.tick_params(axis="y", colors=color_nci)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.7)

    ax1.set_ylim(-0.4, 0.8)

    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    ax2.plot([], [])
    ax2.tick_params(axis='x', colors=color_final, labelsize=14)
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.set_xlabel("Opinion Value", fontsize=17, color=color_final, labelpad=10)
    ax2.set_ylabel("Density (Opinion Distribution)", fontsize=17, color=color_final, labelpad=10)
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis="y", colors=color_final, labelsize=15)
    ax2.set_ylim(0, 1.2)

    node_initial_opinions = []
    node_final_opinions = []

    for node in G.nodes():
        if node in history_opinions:
            opinion_list = history_opinions[node]
            node_initial_opinions.append(opinion_list[0])
            node_final_opinions.append(opinion_list[-1])

    sns.kdeplot(node_initial_opinions, fill=True, bw_adjust=1.5, color=color_initial, alpha=0.7, label="Initial", ax=ax2)
    sns.kdeplot(node_final_opinions, fill=True, bw_adjust=1.5, color=color_final, alpha=0.7, label="Final", ax=ax2)

    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.7), fontsize=12)

    fig.tight_layout(pad=2.0)
    plt.show()
    plt.savefig(file_path)
    plt.savefig(file_path.replace(".png", ".pdf"))
    plt.close()
    plot_evaluation_results(result_data,  model_type, network_type)

def plot_evaluation_results(result_data,  model_type, network_type):
    steps = range(result_data['final_step'] + 2)

    # Create a figure with 3 rows and 2 columns (3x2 layout)
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # Plot NCI scores in the first subplot (axes[0, 0])
    axes[0, 0].plot(steps, result_data['scores_nci'], label='NCI', marker='o')
    axes[0, 0].set_title('Neighbor Correlation Index (NCI)')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True)

    # Plot Polarization scores in the second subplot (axes[0, 1])
    axes[0, 1].plot(steps, result_data['scores_polarization'], label='Polarization', marker='s', color='orange')
    axes[0, 1].set_title('Polarization')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].grid(True)

    # Plot Global Disagreement scores in the third subplot (axes[1, 0])
    axes[1, 0].plot(steps, result_data['scores_gd'], label='Global Disagreement', marker='x', color='green')
    axes[1, 0].set_title('Global Disagreement')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(True)

    # Leave the last subplot (axes[2, 1]) empty or use for additional information if needed
    axes[2, 1].axis('off')  # Turn off the axis for an empty subplot

    # Adjust layout and set the overall title
    if network_type == "small_world":
        network_name = "Small World Network"
    elif network_type == "scale_free":
        network_name = "Scale Free Network"
    else:
        network_name = "Random Network"

    if model_type == "dg":
        model_name = "DeGroot"
    elif model_type == "fj":
        model_name = "FJ"
    else:
        model_name = "BCM"

    fig.suptitle(f"{network_name}_{model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the combined plot

    # Show the plot if needed
    plt.show()


def clear_cache():
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")