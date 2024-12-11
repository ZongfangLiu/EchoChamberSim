import openai
from model import World
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import os
import random
import numpy as np


def set_seed(seed: int):
    """
    Set seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed to set for random number generation.
    """
    # Set the seed for random library
    random.seed(seed)

    # Set the seed for numpy
    np.random.seed(seed)

    print(f"Seed set to {seed} for random, numpy, torch (CPU/GPU)")


if __name__ == "__main__":
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
    parser.add_argument("--gpt_temp", type=float, default=0.5, help="temperature for gpt, higher means more diversity")
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
    parser.add_argument(
        "--mitigation_perspectives_only",
        type=lambda x: x.lower() == 'true',
        default=False,
        help="whether to give a opposite opinion"
    )


    args = parser.parse_args()

    print(f"Parameters: {args}")
    args.exp_name = f"agents_{args.num_agents}_reco_{args.recommendation}_inter_{args.max_interactions}_temp_{args.gpt_temp}_seed_{args.seed}"
    if args.mitigation_step != 1000:
        args.exp_name += f"_mitigation_step_{args.mitigation_step}"
    if not args.with_long_memory:
        args.exp_name += f"_without_long_memory"
    if args.mitigation_perspectives_file is not None:
        args.exp_name += "_with_mitigation_perspectives"
    if args.mitigation_perspectives_only:
        args.exp_name += "_only"
    args.exp_dir = f"experiments_{args.gpt_model}/{args.network_type}/{args.topic}"

    if args.gpt_model == "gpt-4o-mini":
        args.gpt_model = "gpt-4o-mini-2024-07-18"

    leaders = [10, 30]
    # leaders = [8, 20]
    leader_opinions = [-1, 1]

    # Set your seed
    seed = args.seed
    set_seed(seed)

    # Ensure output and checkpoint directories exist
    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")

    for i in range(args.load_from_run, args.no_of_runs):
        print(f"--------Run - {i + 1}---------")
        checkpoint_path = f"checkpoint/run-{i + 1}"
        output_path = f"output/run-{i + 1}"

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if args.load_from_run != 0:  # Load specific checkpoint from the specified run
            checkpoint_file = f"checkpoint/run-{args.load_from_run + 1}/{args.name}-{args.offset}.pkl"
            if os.path.exists(checkpoint_file):
                model = World.load_checkpoint(checkpoint_file)
            else:
                print(f"Warning! Checkpoint not found. Initializing new world for run {args.load_from_run + 1}.")
                model = World(num_agents=args.num_agents, leaders=leaders, network_type=args.network_type,
                              max_interactions=args.max_interactions, belief_keywords_file=args.belief_keywords_file,
                              exp_name=args.exp_name, exp_dir=args.exp_dir, load_network=args.load_network,
                              gpt_model=args.gpt_model, mitigation_step=args.mitigation_step,
                              with_long_memory=args.with_long_memory,
                              mitigation_perspectives_file=args.mitigation_perspectives_file,
                              mitigation_perspectives_only=args.mitigation_perspectives_only,
                              temp=args.gpt_temp, topic=args.topic, recommendation=args.recommendation, seed=seed)
        else:
            model = World(num_agents=args.num_agents, leaders=leaders, network_type=args.network_type,
                          max_interactions=args.max_interactions, belief_keywords_file=args.belief_keywords_file,
                          exp_name=args.exp_name, exp_dir=args.exp_dir, load_network=args.load_network,
                          gpt_model=args.gpt_model, mitigation_step=args.mitigation_step,
                          with_long_memory=args.with_long_memory,
                          mitigation_perspectives_file=args.mitigation_perspectives_file,
                          mitigation_perspectives_only=args.mitigation_perspectives_only,
                          temp=args.gpt_temp, topic=args.topic, recommendation=args.recommendation, seed=seed)

        model.run_model(args.step_count)
        data = model.datacollector.get_model_vars_dataframe()

        df = pd.DataFrame(data)

        # Insert a step column
        df.insert(0, 'Step', range(0, len(df)))

        # Save data
        df.to_csv(output_path + f"/{args.name}-data.csv")

        # Plot and save required figures for each run
        plt.figure(figsize=(10, 6))
        plt.plot(df['Step'], df['Polarization'], label="Polarization")
        plt.plot(df['Step'], df['NeighborCorrelationIndex'], label="Neighbor Correlation Index")
        plt.xlabel('Step')
        plt.ylabel('Metric Value')
        plt.title('Polarization and Neighbor Correlation Index Over Time')
        plt.legend()
        plt.savefig(output_path + f'/{args.name}-metrics.png', bbox_inches='tight')

        # Save final checkpoint after successful run
        model.save_checkpoint(file_path=checkpoint_path + f"/{args.name}-completed.pkl")
