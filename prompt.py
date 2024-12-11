#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   prompt.py
@Contact :   zongfang.liu@mbzuai.ac.ae
@License :   (C)Copyright 2022-2023, ZongfangLiu

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2024/9/13 16:08   zfliu      1.0         None
"""


update_opinion_prompt = (
    """
    Your previous opinion: {opinion}
    Your previous belief value: {belief}
    Your long-term memory: {long_mem}
    Belief values: '-2' for strongly oppose, '-1' for somewhat oppose, '0' for neutral, '1' for somewhat support, '2' for strongly support.

    Task:
    Reflect on your opinion and belief, considering whether to maintain your stance or adjust it based on your long-term memory.

    Instructions:
    - Think like a human: Decide whether to hold firm in your own opinion or adapt based on the influence of the opinions you have heard.

    Output structure (in code format):

    opinion: Provide your current opinion on the topic '{topic}' in several sentences. Your opinion must contain one keyword from {belief_keywords} that reflects your stance. It should begin with: "I {{the selected keyword}}"

    belief: Indicate your current belief value regarding the topic.

    reasoning: Explain the reasoning behind your opinion and belief, elaborating on whether you upheld your original stance or were influenced by the opinions in your long-term memory.
    """
)

update_opinion_prompt_no_reasoning = (
    """
    Your previous opinion: {opinion}
    Your previous belief value: {belief}
    Your long-term memory: {long_mem}
    Belief values: '-2' for strongly oppose, '-1' for somewhat oppose, '0' for neutral, '1' for somewhat support, '2' for strongly support.

    Task:
    Reflect on your opinion and belief, considering whether to maintain your stance or adjust it based on your long-term memory.

    Instructions:
    - Think like a human: Decide whether to hold firm in your own opinion or adapt based on the influence of the opinions you have heard.

    Output structure (in code format):

    opinion: Provide your current opinion on the topic '{topic}' in several sentences. Your opinion must contain one keyword from {belief_keywords} that reflects your stance. It should begin with: "I {{the selected keyword}}"

    belief: Indicate your current belief value regarding the topic.
    """
)


init_opinion_prompt = (
    """
    Given the topic '{topic}', your belief value is {belief}. Please provide your opinion, ensuring you include the keyword '{keyword}' to reflect your stance. 
    Also, explain the reasoning behind your opinion.

    Belief Values: Use the following scale to indicate your belief:
    • '-2' for firmly reject,
    • '-1' for somewhat disagree,
    • '0' for neutral or undecided,
    • '1' for somewhat agree,
    • '2' for strongly support.

    Output Structure (in code format):

    opinion: Your opinion on '{topic}' with clear focus on your stance. It should begin with: "I {keyword} ..."
    Belief: {belief}
    Reasoning: The reasoning behind your opinion and belief. It should be less than 150 words.
    """
)

system_prompt_template = (
    """
    Imagine you are a human. Your name is {name}, and your gender is {gender}. 
    You are {age} years old. Your personality is shaped by these specific traits: {traits}. 
    Your educational background is at the level of {qualification}.
    Act according to this human identity, letting these details fully define your thoughts, responses, interactions, and decisions.
    """
)


system_prompt_leader_template = (
    """
    Imagine you are a human. Your name is {name}, and your gender is {gender}. 
    You are {age} years old. Your personality is shaped by specific traits {traits}. 
    You have an educational background at the level of {qualification}.
    You are acting this human identity, with these details, fully defines your thoughts, responses, word usage, interactions and decisions.
    As an information distributor, you must firmly hold to your own opinions and refrain from adopting the views of others.
    """
)

reflecting_prompt = (
    """
    The opinions you have heard so far: {opinions}

    Task:
    Summarize the opinions provided to form your short-term memory.

    Instructions:
    - Do not add or create information that is not present in the provided opinions.
    - Start the summary with: "In my short-term memory, ..."
    - Provide a brief and accurate summary of the opinions shared with you.

    Output:
    - short_term_memory: Your summarized short-term memory statement.
    """
)

long_memory_prompt = (
    """
    Recap of Previous Long-Term Memory: {long_memory}
    Today's Short-Term Memory: {short_memory}

    Task:
    Using only the information in the previous long-term memory and today's short-term memory, create an updated long-term memory.

    Instructions:
    - Do not introduce any new information that is not present in the provided memories.
    - Start the updated memory with: "In my long-term memory, ..."
    - Accurately combine key details from both the long-term and short-term memories into a clear summary.

    Output:
    - long_term_memory: Your new, consolidated long-term memory statement.
    """
)
