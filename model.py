import mesa
from agent import SocialAgent
from network import generate_network
from datetime import datetime, timedelta
from mesa.datacollection import DataCollector
import dill as pickle
from tqdm import tqdm
from utils import *
from prompt import *


def load_network_structure(file_path):
    with open(file_path, "r") as file:
        network_data = json.load(file)

    G = nx.Graph()

    G.add_nodes_from(network_data["nodes"])

    G.add_edges_from(network_data["edges"])

    return G


class World(mesa.Model):
    def __init__(self, num_agents, leaders, gpt_model, network_type="scale_free", load_network=False,
                 belief_keywords_file=None, exp_name="default_exp",
                 mitigation_perspectives_file=None,
                 exp_dir="./experiments", mitigation_step=1000, with_long_memory=True,
                 mitigation_perspectives_only=False,
                 topic="euthanasia", temp=0.5, recommendation="random", seed=50, **kwargs):
        super().__init__()
        self.belief_keywords = self.load_belief_keywords_file(belief_keywords_file)
        self.num_agents = num_agents
        self.network_type = network_type
        self.step_count = kwargs.get('step_count', 2)
        self.name = exp_name
        self.run_dir = os.path.join(exp_dir, self.name)
        self.temp = temp
        self.recommendation = recommendation
        self.backgrounds = {}
        self.leaders = leaders
        self.leaders_pos = {}
        self.mitigation_step = mitigation_step
        self.with_long_memory = with_long_memory
        self.mitigation_perspectives_only = mitigation_perspectives_only

        self.gpt_model = gpt_model

        os.makedirs(self.run_dir, exist_ok=True)
        self.opinion_templates = self.load_opinions()
        self.topic = self.opinion_templates[topic]

        self.current_date = datetime(2024, 1, 1)
        self.max_interactions = kwargs.get('max_interactions', 5)

        if mitigation_perspectives_file is not None:
            with open(mitigation_perspectives_file, 'r') as json_file:
                self.mitigation_perspectives = json.load(json_file)["perspectives"]
                print(self.mitigation_perspectives)
        else:
            self.mitigation_perspectives = mitigation_perspectives_file

        network_file = f"./data/{network_type}_network_num_agents_{num_agents}_seed_{seed}.json"
        belief_file = f"./data/numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}.json"

        if self.gpt_model == "gpt-4o-mini-2024-07-18":
            gpt_model_name = "gpt-4o-mini"
        else:
            gpt_model_name = self.gpt_model

        backgrounds_file = f"./data/agents_backgrounds_num_agents_{num_agents}_{topic}_{gpt_model_name}.json"
        if load_network and network_file:
            self.G = load_network_structure(network_file)
        else:
            self.G = generate_network(network_type, num_agents, **kwargs)
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)

        if os.path.exists(belief_file):
            with open(belief_file, 'r') as json_file:
                data = json.load(json_file)
            beliefs = {key: int(value * 2) for key, value in data.get('opinions', {}).items()}
        else:
            beliefs = self.generate_balanced_beliefs(self.num_agents)
            random.shuffle(beliefs)

        if os.path.exists(backgrounds_file):
            self.backgrounds = self.load_backgrounds(backgrounds_file)
        else:
            self.backgrounds = self.create_and_save_backgrounds(self.num_agents, self.leaders, beliefs,
                                                                backgrounds_file)

        self.datacollector = DataCollector(
            model_reporters={
                "Polarization": self.compute_polarization,
                "NeighborCorrelationIndex": self.compute_nci,
                "EchoChamberEffect": self.compute_echo_chamber_effect,
            },
            agent_reporters={"Opinion": "opinion"}
        )

        for i, node in enumerate(tqdm(self.G.nodes(), desc="Creating agents")):
            agent = self.create_agent(i, beliefs)
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)
            if node in leaders:
                self.leaders_pos[node] = agent.pos
                print(f"Pos of leader {node}: {agent.pos}")

    def create_and_save_backgrounds(self, num_agents, leaders, beliefs, file_path):
        backgrounds = {}

        for i in tqdm(range(num_agents), desc="Initializing agents"):
            name = f"Agent_{i}"
            age = random.randint(18, 65)
            qualification = self.generate_qualification()
            traits = generate_big5_traits(1)
            gender = self.generate_gender()

            system_prompt_temp = system_prompt_template.format(name=name, gender=gender,
                                                               age=age,
                                                               traits=traits,
                                                               qualification=qualification)

            backgrounds[str(i)] = {
                "name": name,
                "age": age,
                "education level": qualification,
                "traits": traits,
                "gender": gender,
                "system_prompt": None,
                "initial_opinion": None,
                "initial_reasoning": None
            }

            initial_opinion, initial_reasoning = self.generate_initial_opinion_and_reasoning(system_prompt_temp,
                                                                                             beliefs[str(i)],
                                                                                             self.gpt_model)

            system_prompt = system_prompt_temp

            if i in leaders:
                system_prompt += f"""You are an information distributor, you must firmly hold to your own opinion {initial_opinion} and refrain from adopting the views of others."""
                print(system_prompt)

            backgrounds[str(i)]["initial_opinion"] = initial_opinion
            backgrounds[str(i)]["initial_reasoning"] = initial_reasoning
            backgrounds[str(i)]["system_prompt"] = system_prompt

        with open(file_path, "w") as file:
            json.dump({"backgrounds": backgrounds}, file, indent=4)

        return backgrounds

    def load_backgrounds(self, file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return data["backgrounds"]

    def create_agent(self, i, beliefs):

        background_item = self.backgrounds[str(i)]
        name = background_item["name"]
        age = background_item["age"]
        qualification = background_item["education level"]
        traits = background_item["traits"]
        gender = background_item["gender"]
        initial_belief = beliefs[str(i)]
        initial_opinion = background_item["initial_opinion"]
        initial_reasoning = background_item["initial_reasoning"]
        system_prompt = background_item["system_prompt"]

        agent = SocialAgent(model=self,
                            unique_id=i,
                            name=name,
                            gender=gender,
                            age=age,
                            traits=traits,
                            qualification=qualification,
                            initial_belief=initial_belief,
                            belief_keywords=self.belief_keywords,
                            initial_opinion=initial_opinion,
                            initial_reasoning=initial_reasoning,
                            system_prompt=system_prompt,
                            gpt_model=self.gpt_model,
                            temp=self.temp,
                            mitigation_perspectives=None,
                            with_long_memory=self.with_long_memory,
                            topic=self.topic)

        return agent

    def generate_balanced_beliefs(self, num_agents):

        belief_values = [-2, -1, 1, 2]

        beliefs_per_value = num_agents // len(belief_values)
        extra = num_agents % len(belief_values)

        beliefs = []
        for value in belief_values:
            beliefs.extend([value] * beliefs_per_value)

        if extra > 0:
            additional_beliefs = random.choices(belief_values, k=extra)
            beliefs.extend(additional_beliefs)

        return beliefs

    def compute_echo_chamber_effect(self):

        total_similarity = 0
        total_connections = 0

        for agent in self.schedule.agents:
            neighbors = self.grid.get_neighbors(agent.pos, include_center=False)
            agent_belief = agent.belief

            for neighbor in neighbors:
                neighbor_belief = neighbor.belief
                similarity = 1 - abs(agent_belief - neighbor_belief)
                total_similarity += similarity
                total_connections += 1

        if total_connections == 0:
            return 0

        echo_chamber_index = total_similarity / total_connections
        return echo_chamber_index


    def load_opinions(self):
        with open("opinions.json", "r") as file:
            return json.load(file)

    def load_belief_keywords_file(self, belief_keywords_file):
        with open(belief_keywords_file, "r") as file:
            keywords_file = json.load(file)
        return keywords_file

    def generate_initial_opinion_and_reasoning(self, system_prompt, initial_belief, gpt_model):

        belief_str = str(initial_belief)
        # with open(self.belief_keywords_file, "r") as file:
        #     keyword_file = json.load(file)
        belief_keywords = self.belief_keywords

        system_msg = system_prompt
        user_msg = init_opinion_prompt.format(topic=self.topic, belief=initial_belief,
                                              keyword=random.choice(belief_keywords[belief_str]))

        # initial temp is high.
        response = get_completion_from_messages_structured(system_messages=system_msg, messages=user_msg,
                                                           model=gpt_model,
                                                           temperature=0.7, response_type=update_opinion_response)
        tweet = response.opinion
        reasoning = response.reasoning

        return tweet, reasoning

    def generate_qualification(self):
        qualifications = ["No Education", "High School", "Bachelor's Degree", "Master's Degree", "PhD"]
        return random.choice(qualifications)

    def generate_traits(self):
        traits = ["Extroverted", "Introverted", "Cautious", "Risk-taking", "Analytical", "Emotional", "Assertive",
                  "Flexible"]
        return random.choice(traits)

    def generate_gender(self):
        genders = ['male', 'female']
        return random.choice(genders)

    def save_network_structure(self):
        clustering_coefficient = nx.average_clustering(self.G)
        avg_path_length = nx.average_shortest_path_length(self.G) if nx.is_connected(self.G) else None
        density = nx.density(self.G)
        diameter = nx.diameter(self.G) if nx.is_connected(self.G) else None

        network_data = {
            "nodes": list(self.G.nodes),
            "edges": list(self.G.edges),
            "clustering_coefficient": clustering_coefficient,
            "average_path_length": avg_path_length,
            "density": density,
            "diameter": diameter
        }

        file_path = os.path.join(self.run_dir, "network_structure.json")
        with open(file_path, "w") as file:
            json.dump(network_data, file, indent=4)

        plt.figure(figsize=(8, 8))
        nx.draw(self.G, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', font_size=10)
        plt.title(f"Network Structure for {self.name}")
        plt_path = os.path.join(self.run_dir, "network_structure.png")
        plt.savefig(plt_path)
        plt.close()

    def save_agents_data(self, file_path):
        agents_data = {}

        for agent in self.schedule.agents:
            agents_data[agent.unique_id] = {
                "opinions": agent.opinions,
                "beliefs": agent.beliefs,
                "reasonings": agent.reasonings,
                "short-memory": agent.short_memory_full,
                "long_memory": agent.long_memory_full,
            }

        with open(file_path, "w") as file:
            json.dump(agents_data, file, indent=4)

    def save_model_data(self):
        model_data = {
            "step": self.schedule.time,
            "date": str(self.current_date),
            "polarization": self.compute_polarization(),
            "neighbor_correlation_index": self.compute_nci()
        }

        file_path = os.path.join(self.run_dir, "model_overview.json")
        with open(file_path, "a") as file:
            json.dump(model_data, file)
            file.write("\n")

    def compute_polarization(self):
        beliefs = [agent.belief for agent in self.schedule.agents]
        polarization_index = sum(beliefs) / len(beliefs)
        return polarization_index

    def compute_nci(self):
        return sum([agent.belief == neighbor.belief for agent in self.schedule.agents
                    for neighbor in self.grid.get_neighbors(agent.pos)]) / self.num_agents

    def decide_agent_interactions(self, recommendation='random'):
        for agent in self.schedule.agents:

            neighbors = self.grid.get_neighbors(agent.pos)
            agent.agent_interaction = []

            # To do: more recommendation algorithms
            if recommendation == 'random':
                random.shuffle(neighbors)
                neighbors = neighbors[:self.max_interactions]
            elif recommendation == 'similarity':
                neighbors_selected = []
                for neighbor in neighbors:
                    if abs(agent.belief - neighbor.belief) <= 2:
                        neighbors_selected.append(neighbor)
                neighbors = neighbors_selected
            else:
                neighbors_selected = []
                for neighbor in neighbors:
                    if abs(agent.belief - neighbor.belief) >= 2:
                        neighbors_selected.append(neighbor)
                neighbors = neighbors_selected

            for neighbor in neighbors:

                neighbor_agent = neighbor
                agent.agent_interaction.append(neighbor_agent)

    def step(self):
        self.decide_agent_interactions(recommendation=self.recommendation)
        print(f"start step: {self.schedule.time}")
        if self.schedule.time >= self.mitigation_step:
            previous_status_leaders = [self.grid.get_cell_list_contents([pos])[-1] for pos in self.leaders_pos]
            print(previous_status_leaders)
            for agent in self.schedule.agents:
                agent.mitigation_perspectives = None
                if agent.belief in [-2, 2] and agent.pos not in self.leaders_pos:
                    # print(self.grid.get_cell_list_contents([agent.pos]))
                    # agent.agent_interaction.append(self.grid.get_cell_list_contents([agent.pos])[-1])
                    # print(previous_status_leaders)
                    agent.mitigation_perspectives = self.mitigation_perspectives
                    for previous_status_leader in previous_status_leaders:
                        if previous_status_leader.belief != agent.belief:
                            # print("Original:", agent.agent_interaction)
                            if not self.mitigation_perspectives_only:
                                agent.agent_interaction.append(previous_status_leader)
                            # print("After adding:", agent.agent_interaction)
                            # print(f"Cur agent's belief: {agent.belief}")
                            # print(f"Oppsite leader's belief: {previous_status_leader.belief}")

        self.schedule.step()
        for agent in self.schedule.agents:
            # agent.mitigation_perspectives = None
            update_day(agent)

        self.datacollector.collect(self)
        self.current_date += timedelta(days=1)

        agents_file_path = os.path.join(self.run_dir, f"agents_interaction_data.json")
        self.save_agents_data(agents_file_path)

        self.save_model_data()


    def run_model(self, step_count):
        for _ in tqdm(range(step_count), desc="Running Model"):
            self.step()
            print(
                f"Current date: {self.current_date}, Polarization: {self.compute_polarization()}, Echo Chamber Effect: {self.compute_echo_chamber_effect()}")

        self.save_checkpoint(os.path.join(self.run_dir, f"{self.name}_checkpoint.pkl"))
        agents_file_path = os.path.join(self.run_dir, "agents_data.json")
        self.save_agents_data(agents_file_path)

    def save_checkpoint(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_checkpoint(file_path):

        with open(file_path, "rb") as file:
            return pickle.load(file)
