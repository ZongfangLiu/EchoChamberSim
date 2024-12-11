#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Contact :   zongfang.liu@mbzuai.ac.ae
@License :   (C)Copyright 2022-2023, ZongfangLiu

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2024/9/13 16:08   zfliu      1.0         None
"""
from names_dataset import NameDataset
import numpy as np
import random
#from openai import OpenAI
import openai
import json
from pydantic import BaseModel
from prompt import *
import time
import os
import shutil

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.stats import pearsonr


client = openai.Client(
    # This is the default and can be omitted
    api_key="xxxx",
    base_url="xxx"
)


def metric_neighbors_correlation_index(G, opinions):
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

    polarization = np.sum((np.array(opinion_values) - mean_opinion) ** 2)

    return polarization


def metric_global_disagreement(G, opinions):

    global_disagreement = 0

    for i in G.nodes():
        local_disagreement = 0
        for j in G.neighbors(i):
            weight_ij = G[i][j].get('weight', 1)
            opinion_diff = opinions[i] - opinions[j]
            local_disagreement += weight_ij * (opinion_diff ** 2)

        global_disagreement += local_disagreement

    global_disagreement *= 0.5

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

    time_steps = len(next(iter(agent_data.values()))["beliefs"])  # 获取时间步总数
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
        pos = nx.kamada_kawai_layout(G)

    node_colors = [opinions[str(node)] for node in G.nodes()]

    node_size = [G.degree[n] * 100 for n in G.nodes()]

    # Define color map and normalization range for opinions
    # cmap = plt.get_cmap('coolwarm_r')  # You can experiment with other colormaps like 'plasma' or 'viridis'
    cmap = plt.get_cmap('coolwarm')
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
    plt.savefig(file_path)
    plt.savefig(file_path.replace(".png", ".pdf"))
    plt.show()
    plt.close()


def clear_cache():
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")