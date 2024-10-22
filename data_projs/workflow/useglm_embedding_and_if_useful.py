#使用glm进行emmbedding，然后存储向量和ifuseful。都写到一个文件里面
import random
import if_useful as if_useful
import os
import json
import time
import asyncio


#api_qaa.T_junctions_embedding_init()



import pandas as pd


import csv 

def write_time_log(log_list, file_name='embedding_time_log.csv'):
    # Ensure the input is a list
    if not isinstance(log_list, list):
        raise ValueError("Input must be a list")
    
    # Write to a CSV file
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(log_list)

input_path = 'summarized_result_40_turbo'

output_path = 'useglm_embedding_and_if_useful_output'
output_file = output_path + '//' + 'embedding_and_if_useful.json'
#output_file = output_path + '//' + 'embedding_and_if_useful.csv'
time_log_file = output_path + '//' + 'embedding_time_log.csv'

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

from transformers import AutoTokenizer, AutoModel
import torch

def glm_init():
    global tokenizer,model
    tokenizer = AutoTokenizer.from_pretrained("D:\\models\\glm_2\\ChatGLM2-6B-main\\THUDM\\chatglm2-6b", trust_remote_code=True)
    #model = AutoModel.from_pretrained("/home/user/imported_models/chatglm2-6b/huggingface/THUDM/chatglm2-6b", trust_remote_code=True).half().to("cuda:1").eval()
    model = AutoModel.from_pretrained("D:\\models\\glm_2\\ChatGLM2-6B-main\\THUDM\\chatglm2-6b", trust_remote_code=True).half().cuda().eval()
    return
    


data_path = "output_include_dripping_or_jetting"
output_path = "output_paras"

def get_glm_embedding(text, device="cuda:0"):
    global tokenizer,model
    with torch.no_grad():  # 确保在不需要计算梯度时使用
        inputs = tokenizer([text], return_tensors="pt").to(device)
        resp = model.transformer(**inputs, output_hidden_states=True)
        y = resp.last_hidden_state
        y_mean = torch.mean(y, dim=0, keepdim=True)
        # 删除不再需要的变量
        del inputs, resp, y
        torch.cuda.empty_cache()  # 清理缓存
        return y_mean.cpu().detach().numpy()
        
def deal_with_one_file_async(i,k):
    global input_path,output_file,input_list,NA_dict
    file_name = input_path + '//' + i #.json后缀的名称
    print(i)
    f = open(file_name, 'r', encoding = 'utf-8')
    json_str_list = f.readlines() #所有的
    f.close()
    
    #output_file = output_path + '//' + i
    
    '''
    f_out = open(output_file, 'w', encoding ='utf-8')
    f_out.close() #用的时候再a
    '''

        
    j = 0
    k_this_task = k
    
    for json_str in json_str_list:
    
        
        
        data_dict = json.loads(json_str[:-1]) #最后换行符删掉
        content = data_dict['content']
        
        max_token = 4000
        if_write_log = 0
        start_time = time.time()
        
        if len(content) < max_token:
            
            #比对这个content是否存在
            
            if content in content_out_list:
                print('已经生成过！')
                
            else:
            
                if_write_log = 1
                
                #开始处理
                print(f'处理第{k_this_task}篇文献的第{j}//{len(json_str_list)}部分')
                embedding = get_glm_embedding(content)
                #print(len(embedding),len(embedding[0]))
                if_useful_rst = if_useful.judge_if_useful(data_dict["summarize_result"])

                output_dict = {'if_useful':if_useful_rst, 'embedding':embedding.tolist()[0][0], 'content':data_dict['content']}.copy()

                
                #print(f"writting:f_out:{output_file[0:50]}")
                #append_df_to_csv(df_new, csv_file_path)
                f_out = open(output_file, 'a', encoding ='utf-8')
                f_out.write(json.dumps(output_dict))
                f_out.write('\n')
                f.close()
                #print(f"down:f_out:{output_file[0:50]}")
                
        else: #太长
            content_list = []
            content_remain = content #切剩下的
            while len(content_remain) > max_token:
                content_list.append(content_remain[0:max_token])
                content_remain = content_remain[max_token:]
            for content_in_list in content_list:
            
                if content in content_out_list:
                    print('已经生成过！')
                    
                else:
                    #开始处理
                    
                    if_write_log = 1
                    
                    embedding = get_glm_embedding(content)
                    if_useful_rst = if_useful.judge_if_useful(data_dict["summarize_result"])

                    output_dict = {'if_useful':if_useful_rst, 'embedding':embedding.tolist()[0][0], 'content':data_dict['content']}.copy()
                    
                    f_out = open(output_file, 'a', encoding ='utf-8')
                    f_out.write(json.dumps(output_dict))
                    f_out.write('\n')
                    f.close()
        
        end_time = time.time()  # end time
        elapsed_time = end_time - start_time  # calculate elapsed time
        if if_write_log == 1:
            print(f'Time taken to process part {j} of {k_this_task}th document: {elapsed_time:.2f} seconds')
            write_time_log([k_this_task, j, len(content), elapsed_time],file_name=time_log_file)
            
        j += 1

        
    #f_out.close()
    
'''
k = -1
def main():
    global k
    for iii in input_list:
        k += 1
        deal_with_one_file(iii)


main()
'''
def main():

    try:
        content_out_list = [] #用于比对是否生成过
        f_out = open(output_file, 'r', encoding ='utf-8')
        json_str_list_out = f_out.readlines()
        for json_str in json_str_list_out:
            data_dict = json.loads(json_str[:-1]) #最后换行符删掉
            content = data_dict['content']
            content_out_list.append(content)
        f_out.close()
    except:
        content_out_list = []
            
            
    glm_init()
    print('载入成功')
    
    input_list = os.listdir(input_path) 
    kt = 0
    #asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) 
    #loop = asyncio.get_event_loop()
    task_list = []
    #print("第143个",input_list[143])
    part = 5
    #for iii in input_list[0:(part + 1)*20]:
    for iii in input_list:
        #task_list.append(loop.create_task(deal_with_one_file_async(iii,kt)))
        deal_with_one_file_async(iii,kt)
        kt += 1
        
    #print("task_list,",len(task_list))
    #print(task_list)

    #loop.run_until_complete(asyncio.wait(task_list))