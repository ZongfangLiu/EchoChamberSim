#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   visualization.py    
@Contact :   zongfang.liu@mbzuai.ac.ae
@License :   (C)Copyright 2022-2023, ZongfangLiu

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2024/9/13 16:08   zfliu      1.0         None
"""
import argparse
import os

from utils import generate_belief_animation, visulize_opinions

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

parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini", help="topic for agents")
parser.add_argument("--mitigation_step", type=int, default=1000, help="the time to start mitigation")
parser.add_argument(
    "--with_long_memory",
    type=lambda x: x.lower() == 'true',
    default=True,
    help="whether to use long term memory, for ablation study"
)
parser.add_argument("--mitigation_perspectives_file", type=str, default=None, help="")

args = parser.parse_args()

print(f"Parameters: {args}")
args.max_interactions = -1
# for args.network_type in ["small_world", "scale_free", "random"]:
for args.network_type in ["scale_free"]:
    # for args.recommendation in ["random", "similarity"]:
    for args.recommendation in ["similarity"]:
    # for args.recommendation in ["random"]:
        # for args.topic in ['ai', 'euthanasia', 'lifespan']:
        for args.topic in ['euthanasia']:

            args.exp_name = f"agents_{args.num_agents}_reco_{args.recommendation}_inter_{args.max_interactions}_temp_{args.gpt_temp}_seed_{args.seed}"
            if args.mitigation_step != 1000:
                args.exp_name += f"_mitigation_step_{args.mitigation_step}"
            if not args.with_long_memory:
                args.exp_name += f"_without_long_memory"
            if args.mitigation_perspectives_file is not None:
                args.exp_name += "_with_mitigation_perspectives"

            args.exp_dir = f"experiments_{args.gpt_model}_formal/{args.network_type}/{args.topic}"
            exp_path = os.path.join(args.exp_dir, args.exp_name)

            network_file_path = f"./data/{args.network_type}_network_num_agents_{args.num_agents}_seed_{args.seed}.json"
            agent_interaction_data_file_path = os.path.join(exp_path, "agents_interaction_data.json")
            output_file_path = os.path.join(exp_path, "opinion_dynamics.gif")
            # print(output_file_path)
            show_label = False

            # generate_belief_animation(network_file_path, agent_interaction_data_file_path,
            #                           output_file_path, args.network_type, show_label=show_label)

            # origin = True
            step = 0
            visulize_opinions(network_file_path, agent_interaction_data_file_path, exp_path, args.recommendation, args.network_type,
                              args.seed, show_label, step)
            # origin = False
            step = 30
            visulize_opinions(network_file_path, agent_interaction_data_file_path, exp_path, args.recommendation, args.network_type,
                              args.seed, show_label, step)



