import os
import json
import time
import asyncio
import api 
import if_useful as iu
from parse_json import *
from train_and_eval import *
from useglm_embedding_and_if_useful import get_glm_embedding,glm_init

# Input: texts
# Output: summarized_result

MAX_NUM = 2 #训练几篇为止
input_path = 'texts'
# Construct the paths
output_path = 'summarized_result\\MAX_NUM_' + str(MAX_NUM)
log_path = 'time_logs\\MAX_NUM_' + str(MAX_NUM)

# Create the directories if they do not exist
for path in [output_path, log_path]:
    if not os.path.exists(path):
        os.makedirs(path)

Raw_file_PATH = r".\data\temp_results\train_data.json" #训练数据
Threshold_max = 0.00002
enough_num = 14 #阈值线性增加到14，因为14篇已经有很好的召回率和精准率


# Check if the output_path directory exists in the current directory
if not os.path.exists(output_path):
    # If it does not exist, create the directory
    os.makedirs(output_path)

input_list = os.listdir(input_path)[:]

import csv

def write_log(log_list, file_name=log_path + '\\workflow_log.csv'):
    # Ensure the input is a list
    if not isinstance(log_list, list):
        raise ValueError("Input must be a list")
    
    # Write to a CSV file
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(log_list)

#def summarize(content):
    #summarize_result = await api.T_junctions_qaa_async_40(content)  # extraction result

def summarize(content, summarize_result_for_check):
    summarize_result = summarize_result_for_check  # extraction result

    #await asyncio.sleep(sleep_time) 
    print(summarize_result[:100])
    
    if_useful = False
    
    try:
        structured_result = json.loads(summarize_result)
        if_useful = iu.judge_if_useful(summarize_result)
    except:  # does not meet JSON format
        print('Incorrect format')
        structured_result = {}

    output_dict = {'if_useful': if_useful, 'summarize_result': summarize_result, 'content': content}
    return output_dict

NA_dict = {"continuous phase":{"fluid composition":"N/A", 
                       "flow rate":"N/A", 
                       "density":"N/A", 
                       "viscosity":"N/A", 
                       "interface tension":"N/A",
                       "channel’s width":"N/A",
                       "channel’s depth":"N/A",
                       "weber number":"N/A",
                       "capillary number":"N/A"},
    "dispersed phase":{"fluid composition":"N/A", 
                       "flow rate":"N/A", 
                       "density":"N/A", 
                       "viscosity":"N/A", 
                       "interface tension":"N/A",
                       "channel’s width":"N/A",
                       "channel’s depth":"N/A",
                       "weber number":"N/A",
                       "capillary number":"N/A"},
    "droplets type":"N/A"}
def na_output(content):
    output_dict = {'if_useful': False, 'summarize_result': NA_dict, 'content': content}
    return output_dict

def linear_increase(value):
    if 0 <= value < enough_num:
        return value * Threshold_max / enough_num
    else:
        return Threshold_max
        
def embedding_summarize_and_train(k, j, content, summarize_result_for_check):
    
    embedding = get_glm_embedding(content)
    if_write_train_data = True
    if_sent_to_summarize_x = False
    
    if k == 0:
    
        output_dict = summarize(content, summarize_result_for_check)
        if_sent_to_summarize_x = True
    
    else:
        
        if j == 0 and k < MAX_NUM:
        
            incremental_training() #重新训练直觉网络
        
        probability = predict_positive_probability_glm(embedding)
        
        threshold = linear_increase(k)

        if probability < threshold: # 没有信息的段落
            
            output_dict = na_output(content)
            
        else:
            
            output_dict = summarize(content, summarize_result_for_check)
            if_sent_to_summarize_x = True
            
            if k > MAX_NUM: #超过这个最大值就不训练了
            
                if_write_train_data = False
                
    if if_write_train_data == True:
        
        #写入训练数据
        train_data_dict = {'if_useful':output_dict['if_useful'], 'embedding':embedding.tolist()[0][0],'no_use_col':'no_use',}.copy()
        train_data_out = open(Raw_file_PATH, 'a', encoding ='utf-8')
        train_data_out.write(json.dumps(train_data_dict))
        train_data_out.write('\n')
        train_data_out.close()
        

        
    return output_dict, if_sent_to_summarize_x      
    
#async def deal_with_one_file_async(i, k):
def deal_with_one_file_async(i, k):
    if_sent_to_summarize = False
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
                #output_dict = embedding_summarize_and_train(k_this_task, j, content)
                #output_dict = embedding_summarize_and_train(k_this_task, j, content, data_dict['summarize_result'])
                output_dict,if_sent_to_summarize = embedding_summarize_and_train(k_this_task, j, content, data_dict['summarize_result'])

                return_str += str(output_dict['summarize_result'])
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
                    #output_dict = embedding_summarize_and_train(k_this_task, j, content, data_dict['summarize_result'])
                    output_dict,if_sent_to_summarize = embedding_summarize_and_train(k_this_task, j, content, data_dict['summarize_result'])
                    
                    return_str += str(output_dict['summarize_result'])
                    with open(output_file, 'a', encoding='utf-8') as f_out:
                        f_out.write(json.dumps(output_dict))
                        f_out.write('\n')
                        
        end_time = time.time()  # end time
        elapsed_time = end_time - start_time  # calculate elapsed time
        
        #for test:
        if if_sent_to_summarize==True:
            elapsed_time += 24.2883
        
        if if_write_log == 1:
            print(f'Time taken to process part {j} of {k_this_task}th document: {elapsed_time:.2f} seconds')
            write_log([k_this_task, j, len(json_str_list), elapsed_time, len(json_str), len(return_str)])

        j += 1


glm_init()

#loop = asyncio.get_event_loop()
task_list = []

# Process in chunks to avoid overloading the system
sleep_time = 5
part = 20
kt = 0
#for iii in input_list[0:(part + 1)*20]:
for iii in input_list:
    #task_list.append(loop.create_task(deal_with_one_file_async(iii, kt)))
    deal_with_one_file_async(iii, kt)
    kt += 1

#print("task_list length:", len(task_list))
#print(task_list)

#loop.run_until_complete(asyncio.wait(task_list))
