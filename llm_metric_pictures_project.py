#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   llm_metric_pictures_project.py
@Contact :   zongfang.liu@mbzuai.ac.ae
@License :   (C)Copyright 2022-2023, ZongfangLiu

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2024/9/13 16:08   zfliu      1.0         None
"""
import argparse
import os
import networkx as nx
import json

from utils_new import generate_belief_animation, visulize_opinions,visulize_metrics,extract_beliefs_at_all_steps,metric_neighbors_correlation_index,metric_polarization, metric_global_disagreement

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="EchoChamberSim", help="Name of the run to save outputs.")
parser.add_argument("--network_type", default="scale_free", choices=["small_world", "scale_free", "random"],
                    help="Type of network structure to use.")
parser.add_argument("--num_agents", default=50, type=int, help="Number of agents in the network.")
parser.add_argument("--step_count", default=30, type=int, help="Total number of steps the simulation should run.")
parser.add_argument("--no_of_runs", default=1, type=int, help="Total number of times you want to run this code.")
parser.add_argument("--offset", default=0, type=int, help="Offset for loading a checkpoint.")
parser.add_argument("--load_from_run", default=0, type=int, help="Specify run # to load checkpoint from.")
parser.add_argument("--max_interactions", default=-1, type=int,
                    help="Maximum number of interactions per agent per step.")
parser.add_argument("--belief_keywords_file", default="./data/belief_keywords.json", type=str,
                    help="JSON file describes the keywords")
# parser.add_argument("--exp_name", default="EU_16_self_temp_0.5_new_keywords_v1", help="Name of experiments")
parser.add_argument("--topic", default="euthanasia", help="topic for agents")
parser.add_argument("--gpt_temp", type=float, default=1.0, help="temperature for gpt, higher means more diversity")
parser.add_argument("--recommendation", type=str, default="random", help="topic for agents")
parser.add_argument("--load_network", type=bool, default=True, help="whether loads existing network structure")
parser.add_argument("--seed", default=50, type=int, help="random seed")

args = parser.parse_args()

print(f"Parameters: {args}")
args.max_interactions = -1
# for args.network_type in ["small_world", "scale_free", "random"]:
for args.network_type in ["scale_free"]:
#     for args.recommendation in ["random", "similarity"]:
    for args.recommendation in ["similarity"]:
#     for args.recommendation in ["random"]:
        # for args.topic in ['ai', 'euthanasia', 'lifespan']:
        for args.topic in ['euthanasia']:
            # args.exp_name = f"agents_{args.num_agents}_reco_{args.recommendation}_inter_{args.max_interactions}_temp_{args.gpt_temp}_seed_{args.seed}"
            # args.exp_dir = f"experiments_gpt-4o-mini_mitigation_formal_v2/{args.network_type}/{args.topic}"
            args.exp_dir = f"experiments_gpt-4o-mini_mitigation_formal_v2/{args.network_type}/{args.topic}"
            # args.exp_dir = f"experiments_gpt-4o-mini_formal/{args.network_type}/{args.topic}"

            directories_in_exp_path = [d for d in os.listdir(args.exp_dir) if os.path.isdir(os.path.join(args.exp_dir, d))]

            for exp_name in directories_in_exp_path:
                exp_path = os.path.join(args.exp_dir, exp_name)

                network_file_path = f"./data/{args.network_type}_network_num_agents_{args.num_agents}_seed_{args.seed}.json"
                agent_interaction_data_file_path = os.path.join(exp_path, "agents_interaction_data.json")
                output_file_path = os.path.join(exp_path, "opinion_dynamics.gif")
                print(agent_interaction_data_file_path)
                # print(output_file_path)
                show_label = False

                #generate_belief_animation(network_file_path, agent_interaction_data_file_path,
                                          # output_file_path, args.network_type, show_label=show_label)

                # origin = True
                step = 0
                visulize_opinions(network_file_path, agent_interaction_data_file_path, exp_path, args.recommendation, args.network_type,
                                  args.seed, show_label, step)
                # # origin = False
                step = 30
                visulize_opinions(network_file_path, agent_interaction_data_file_path, exp_path, args.recommendation, args.network_type,
                                  args.seed, show_label, step)
                step=30
                # visulize_metrics(network_file_path, agent_interaction_data_file_path, exp_path, args.recommendation, args.network_type,
                #                   args.seed, show_label, step)

                with open(network_file_path, "r") as file:
                    network_data = json.load(file)

                G = nx.Graph()
                G.add_nodes_from(network_data["nodes"])
                G.add_edges_from(network_data["edges"])

                beliefs_at_steps = extract_beliefs_at_all_steps(agent_interaction_data_file_path)
                # if step == 0:
                #     opinions = beliefs_at_steps[0]
                #     file_prefix = f"llm_network_{network_type}_model_{model_type}_seed_{network_seed}_origin.png"
                # else:
                #     opinions = beliefs_at_steps[step]
                max_steps = step
                step = 0
                opinions = beliefs_at_steps[step]
                print(opinions)
                print(opinions)
                for node in G.nodes:
                    print(node)

                scores_nci, scores_polarization, scores_gd, scores_rwc = [], [], [], []
                history_opinions = {node: [opinions[f'{node}']] for node in G.nodes()}  # 用于记录每个节点的历史意见
                # metric_opinion_distribution(G, history_opinions)
                score_nci = metric_neighbors_correlation_index(G, opinions,'llm')
                # print('now step: ', step, ', NCI: ', score_nci)
                scores_nci.append(score_nci)

                score_polarization = metric_polarization(G, opinions)
                scores_polarization.append(score_polarization)

                score_gd = metric_global_disagreement(G, opinions,'llm')
                scores_gd.append(score_gd)
                print('here')

                for step in range(max_steps):
                    new_opinions = beliefs_at_steps[step]

                    for node in G.nodes():
                        history_opinions[node].append(new_opinions[f'{node}'])


                    score_nci = metric_neighbors_correlation_index(G, new_opinions,'llm')
                    scores_nci.append(score_nci)

                    score_polarization = metric_polarization(G, new_opinions)
                    scores_polarization.append(score_polarization)

                    score_gd = metric_global_disagreement(G, new_opinions,'llm')
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

                print(f"{args.recommendation}{args.network_type} result polarization: ",
                      round(result_data['scores_polarization'][-1] - result_data['scores_polarization'][0], 4))
                print(f"{args.recommendation}{args.network_type} result gd: ",
                      round(result_data['scores_gd'][-1] - result_data['scores_gd'][0], 4))
                print(f"{args.recommendation}{args.network_type} result: nci",
                      round(result_data['scores_nci'][-1] - result_data['scores_nci'][0], 4))
                import seaborn as sns
                import matplotlib.pyplot as plt


                sns.set(style="whitegrid")
                sns.set_context("talk")

                steps = range(result_data['final_step'] + 2)

                fig, ax1 = plt.subplots(figsize=(8, 8))

                color_palette = sns.color_palette("Set1", n_colors=3)

                ax1.plot(steps, result_data['scores_nci'], label='NCI', marker='o', color=color_palette[0], linewidth=2,
                         markersize=7)
                ax1.set_xlabel('Iteration Days', fontsize=25)
                ax1.set_ylabel('Neighbors Correlation', fontsize=30)
                ax1.tick_params(axis='y', labelsize=18)
                ax1.tick_params(axis='x', labelsize=18)
                ax1.set_ylim(-0.3,0.7)
                ax1.set_xticks(range(0, len(steps), 5))
                ax2 = ax1.twinx()

                ax2.plot(steps, result_data['scores_polarization'], label='Polarization', marker='s',
                         color=color_palette[1], linewidth=2, markersize=7)
                ax2.plot(steps, result_data['scores_gd'], label='Global Disagreement', marker='^', color=color_palette[2],
                         linewidth=2, markersize=7)
                ax2.set_ylabel('Polarization / Global Disagreement', color='k', fontsize=30)
                ax2.tick_params(axis='y', labelcolor='k', labelsize=18)
                ax2.set_ylim(1,3)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()

                ax2.spines['top'].set_color('black')
                ax2.spines['right'].set_color('black')
                ax2.spines['bottom'].set_color('black')
                ax2.spines['left'].set_color('black')

                if args.network_type == "small_world":
                    network_name = "Small World Network"
                elif args.network_type == "scale_free":
                    network_name = "Scale Free Network"
                else:
                    network_name = "Random Network"

                model_name = "LLM"
                recommendation = 'Random recommendation' if args.recommendation == "random" else 'Similarity recommendation'

                fig.subplots_adjust(top=0.85)
                plt.tight_layout()
                plt.savefig(os.path.join(exp_path, exp_name + ".pdf"))

                plt.show()
