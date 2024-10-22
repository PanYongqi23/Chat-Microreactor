import os
import json
import time
import asyncio
import aiohttp
from aiohttp import ClientSession
import api 
import if_useful as iu

# Input: texts
# Output: summarized_result

input_path = 'texts'
output_path = 'summarized_result'

# Check if the output_path directory exists in the current directory
if not os.path.exists(output_path):
    # If it does not exist, create the directory
    os.makedirs(output_path)

input_list = os.listdir(input_path)[:]

import csv

def write_log(log_list, file_name='log.csv'):
    # Ensure the input is a list
    if not isinstance(log_list, list):
        raise ValueError("Input must be a list")
    
    # Write to a CSV file
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(log_list)

async def deal_with_one_file_async(i, k):
    
    global input_path, output_path, input_list
    
    file_name = input_path + '/' + i  # .json file name
    print(i)
    with open(file_name, 'r', encoding='utf-8') as f:
        json_str_list = f.readlines()  # all lines
    
    output_file = output_path + '/' + i
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f_out:
            json_str_list_out = f_out.readlines()
            content_out_list = []  # for checking if generation has occurred
            for json_str in json_str_list_out:
                data_dict = json.loads(json_str[:-1])  # remove the last newline character
                content = data_dict['content']
                content_out_list.append(content)
    except:
        content_out_list = []
        
    j = 0
    k_this_task = k
    
    for json_str in json_str_list:
        
        start_time = time.time()  # get timestamp
        
        if_write_log = 0
        return_str = ''  # for counting the number of returned characters
        
        print(f'Processing the {k_this_task}th document, part {j}/{len(json_str_list)}')
        
        data_dict = json.loads(json_str[:-1])  # remove the last newline character
        content = data_dict['content']
        
        max_token = 128000
        if len(content) < max_token:
            
            # check if this content already exists
            
            if content in content_out_list:
                print('Already generated!')
            else:
                if_write_log = 1
                # start extraction
                summarize_result = await api.T_junctions_qaa_async_40(content)  # extraction result
                await asyncio.sleep(sleep_time) 
                print(summarize_result[:100])
                
                if_useful = False
                
                try:
                    structured_result = json.loads(summarize_result)
                    if_useful = iu.judge_if_useful(summarize_result)
                except:  # does not meet JSON format
                    print('Incorrect format')
                    structured_result = {}

                output_dict = {'if_useful': if_useful, 'summarize_result': summarize_result, 'content': content}
                
                return_str += summarize_result
                with open(output_file, 'a', encoding='utf-8') as f_out:
                    f_out.write(json.dumps(output_dict))
                    f_out.write('\n')
                
        else:  # too long
            content_list = []
            content_remain = content  # remaining content to be cut
            while len(content_remain) > max_token:
                content_list.append(content_remain[:max_token])
                content_remain = content_remain[max_token:]
            for content_in_list in content_list:
                
                if content in content_out_list:
                    print('Already generated!')
                else:
                    
                    if_write_log = 1
                    
                    # start extraction
                    summarize_result = await api.T_junctions_qaa_async_40(content)  # extraction result
                    await asyncio.sleep(sleep_time) 
                    print(summarize_result[:100])
                    
                    if_useful = False
                    
                    try:
                        structured_result = json.loads(summarize_result)
                        if_useful = iu.judge_if_useful(summarize_result)
                    except:  # does not meet JSON format
                        print('Incorrect format')
                        structured_result = {}
                    
                    output_dict = {'if_useful': if_useful, 'summarize_result': summarize_result, 'content': content}
                    
                    return_str += summarize_result
                    with open(output_file, 'a', encoding='utf-8') as f_out:
                        f_out.write(json.dumps(output_dict))
                        f_out.write('\n')
                        
        end_time = time.time()  # end time
        elapsed_time = end_time - start_time  # calculate elapsed time
        if if_write_log == 1:
            print(f'Time taken to process part {j} of {k_this_task}th document: {elapsed_time:.2f} seconds')
            write_log([k_this_task, j, len(json_str_list), elapsed_time, len(json_str), len(return_str)])

        j += 1

loop = asyncio.get_event_loop()
task_list = []

# Process in chunks to avoid overloading the system
sleep_time = 5
part = 20
kt = 0
for iii in input_list[0:(part + 1)*20]:
    task_list.append(loop.create_task(deal_with_one_file_async(iii, kt)))
    kt += 1

print("task_list length:", len(task_list))
print(task_list)

loop.run_until_complete(asyncio.wait(task_list))
