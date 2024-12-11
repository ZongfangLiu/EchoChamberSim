from mesa import Agent
from utils import *
from prompt import *


def get_summary_long(system_prompt, long_memory, short_memory, gpt_model, temp=0.5):

    system_msg = system_prompt
    user_msg = long_memory_prompt.format(long_memory=long_memory, short_memory=short_memory)

    get_summary = get_completion_from_messages_structured(system_messages=system_msg, messages=user_msg,
                                                          model=gpt_model,
                                                          temperature=temp, response_type=long_memory_response).long_term_memory
    # print('long_memory: ', get_summary)

    return get_summary


def get_summary_short(system_prompt, opinions, gpt_model, mitigation_perspectives=None, temp=0.5):

    opinions_text = "\n".join(f"One of your close contacts believes: {opinion}" for opinion in opinions)

    if mitigation_perspectives is not None:
        random.shuffle(mitigation_perspectives)

        for i in range(len(opinions) // 2 + 1):
            opinions_text += f"\n You heard that: {random.sample(mitigation_perspectives, 1)}"

    # user_msg = reflecting_prompt.format(opinions=opinions_text, topic=topic)
    system_msg = system_prompt
    user_msg = reflecting_prompt.format(opinions=opinions_text)
    # print("short term usr message: ", user_msg)

   # msg = [{"role": "user", "content": user_msg}]

    get_summary = get_completion_from_messages_structured(system_messages=system_msg, messages=user_msg,
                                                          model=gpt_model,
                                                          temperature=temp, response_type=reflecting_response).short_term_memory
    # print('short_memory: ', get_summary)

    return get_summary


class SocialAgent(Agent):
    def __init__(self, model, unique_id, name, gender, age, traits, qualification, initial_belief, topic,
                 belief_keywords, gpt_model, temp=0.5, initial_opinion=None, initial_reasoning=None,
                 with_long_memory=True, mitigation_perspectives=None,
                 system_prompt="You are a helpful assistant"):
        super().__init__(unique_id, model)


        self.name = name
        self.gender = gender
        self.age = age
        self.traits = traits
        self.qualification = qualification
        self.topic = topic
        self.belief_keywords = belief_keywords
        self.system_prompt = system_prompt

        self.with_long_memory = with_long_memory

        self.temp = temp
        self.gpt_model = gpt_model

        self.mitigation_perspectives = mitigation_perspectives


        self.initial_opinion = initial_opinion
        self.opinions = [self.initial_opinion]
        self.initial_belief = initial_belief
        self.belief = initial_belief
        self.beliefs = [self.belief]


        self.short_memory_full = []

        self.long_opinion_memory = ""
        self.long_memory_full = [self.long_opinion_memory]

        self.initial_reasoning = initial_reasoning
        self.reasonings = [initial_reasoning]

        self.agent_interaction = []
        self.contact_ids = []

    def interact(self):
        print("I'm agent ", self.unique_id)
        others_opinions = []
        contact_id = []
        # print("Lenth of interaction is:", len(self.agent_interaction))
        for agent in self.agent_interaction:
            contact_id.append(agent.unique_id)
            agent_latest_opinion = agent.opinions[-1]
            others_opinions.append(agent_latest_opinion)

        # self.short_opinion_memory.append(others_opinions)
        self.contact_ids.append(contact_id)

        opinion_short_summary = get_summary_short(self.system_prompt, others_opinions,
                                                  gpt_model=self.gpt_model,
                                                  mitigation_perspectives=self.mitigation_perspectives,
                                                  temp=self.temp)

        self.short_memory_full.append(opinion_short_summary)

        if self.with_long_memory:
            long_mem = get_summary_long(self.system_prompt, self.long_opinion_memory, opinion_short_summary,
                                        gpt_model=self.gpt_model,
                                        temp=self.temp)

            self.long_opinion_memory = long_mem
            self.long_memory_full.append(self.long_opinion_memory)

        self.agent_interaction = []

    def response_and_belief(self, user_msg, gpt_model):

        system_msg = self.system_prompt

        response = get_completion_from_messages_structured(system_messages=system_msg, messages=user_msg,
                                                           model=gpt_model,
                                                           temperature=self.temp, response_type=update_opinion_response)

        tweet = response.opinion
        belief = response.belief
        reasoning = response.reasoning

        return tweet, belief, reasoning

    def step(self):
        '''
        Step function for agent
        '''
        self.interact()

