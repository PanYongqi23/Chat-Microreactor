import openai
import os
from scipy.spatial.distance import cosine
import time
import asyncio

# Retrieve API keys from the accounts file
key_list = []
with open('..//data//accounts.txt', 'r') as f_api:
    str_list = f_api.readlines()
    for i in str_list:
        chat_gpt_key_i = "sk-" + i.split("sk-")[1].rstrip()
        key_list.append(chat_gpt_key_i)

# Instruction to minimize hallucinations
order = 'Answer the question as truthfully as possible using the provided context.'

#os.environ["OPENAI_API_KEY"] = key_list[-1]

from openai import OpenAI,AsyncOpenAI

# Client for chatting async
aclient = AsyncOpenAI(
    base_url="https://gtapi.xiaoerchaoren.com:8932/v1",
    api_key=key_list[-1]
)

# Client for chatting
client = OpenAI(
    base_url="https://gtapi.xiaoerchaoren.com:8932/v1",
    api_key=key_list[-1]
)

# Client for embeddings
aclient_direct = AsyncOpenAI(
    api_key=key_list[0]
)
client_direct = OpenAI(
    api_key=key_list[0]
)

def Chat(prompt1,history=[]):

    messages=[]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": prompt1})
    
    i = 0
    i_max = 100  # Attempt up to 10 times
    while i < i_max:
        try:
            completion = client_direct.chat.completions.create(
                #model="gpt-4-turbo",
                model="gpt-3.5-turbo",
                messages = messages,
                temperature=0,
                stream=True  # this time, we set stream=True
            )
            break
        except Exception as e:  # API failure
            print(e)
    
    if i == i_max:
        print("API error.")
        return

    for chunk in completion:
        token = chunk.choices[0].delta.content
        if token != None:
            yield token

# This function processes a single request and returns a single result
async def Completion_async_40(prompt1):

    i = 0
    i_max = 100  # Attempt up to 10 times
    while i < i_max:
        try:
            completion = await aclient.chat.completions.create(
                model="gpt-4-turbo", 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt1}
                ],
                temperature=0
            )
            break
        except Exception as e:  # API failure
            print(e)
    
    if i == i_max:
        print("API error.")
        return
    
    return completion.choices[0].message.content


async def T_junctions_qaa_async_40(context):
    prompt_T_junctions = ('Please summarize whether the droplets formed are dripping, jetting, or squeezing. '
                          'There may be two phases in the provided context: a continuous phase (usually subscript c) '
                          'and a dispersed phase (usually subscript d). For every phase, please summarize the following '
                          'details in a json object: fluid composition, flow rate or flow velocity (or “r” or "Q"), '
                          'density (or “ρ”), viscosity (or “μ”), interface tension (or “σ”), channel’s width, '
                          'channel’s depth, weber number ("wb"), capillary number (or “Ca”). If multiple sets of '
                          'experiments appear for the same parameter in the text, you should give multiple json objects '
                          'separately. If any information is not provided or you are unsure, use "N/A". Please ignore '
                          'information related to chemical reaction or biological process. In the output json object, '
                          'the parameters of any phase should be represented like this: "continuous phase":{'
                          '"fluid composition":"","flow rate":"","density":"","viscosity":"","interface tension":"","channel’s width":"","channel’s depth":"","channel’s depth":"","weber number":"","capillary number":""}. A complete json object is like this: {"continuous phase":{},"dispersed phase":{},"droplets type":""}. If multiple parameters are provided for the same item, use list like [] to represent them. Please give me the JSON result directly, do not use the Markdown code block. No text explanation is required.\ncontext:')
    prompt_last = prompt_T_junctions + context
    msg = await Completion_async_40(prompt_last)
    return msg

# This function processes embeddings async
async def embedding_async_40(prompt1):
    i = 0
    i_max = 100  # Attempt up to 10 times
    while i < i_max:
        try:
            completion = await aclient_direct.embeddings.create(
                input=prompt1,
                model='text-embedding-3-small'
            )
            break
        except Exception as e:  # API failure
            print(e)
    
    if i == i_max:
        print("API error.")
        return
    
    return completion.data[0].embedding

def embedding_40(prompt1):
    i = 0
    i_max = 100  # Attempt up to 10 times
    while i < i_max:
        try:
            completion = client_direct.embeddings.create(
                input=prompt1,
                model='text-embedding-3-small'
            )
            break
        except Exception as e:  # API failure
            print(e)
    
    if i == i_max:
        print("API error.")
        return
    
    return completion.data[0].embedding
