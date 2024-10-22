#将提取到的结果整成md
import os
import json


input_path = 'summarized_result'
input_list = os.listdir(input_path)

out_put_path = 'summarized_result_md'

error = 0
total = 0
for file in input_list:
    
    file_path = input_path + '//' + file
    
    title = file[:-5]
    
    output_path = out_put_path + '//' + title + '.md'
    
    f_in = open(file_path,'r', encoding = 'utf-8')
    json_str_list = f_in.readlines()
    f_in.close()
    
    f_out = open(output_path,'w',encoding = 'utf-8')
    
    #标题
    f_out.write('# ')
    f_out.write(title)
    f_out.write('\n\n\n\n')
    
    for json_str in json_str_list:
        #每一段文本切片
        total += 1
        data_dict = json.loads(json_str[:-1])
        
        #文本内容
        f_out.write('**文本内容：**')
        f_out.write('\n\n')
        f_out.write(data_dict["content"])
        f_out.write('\n\n')
        
        #提取内容
        info_needed_print = []
        try:
            if data_dict["if_useful"] == True:
            
                #print("\n\nsummarize_result",data_dict["summarize_result"])
                summarize_result = json.loads(data_dict["summarize_result"])

                
                for key in summarize_result: #一般来说这三个key是流动相、分散相和液滴类型
                    None_list = [None,"N/A","",{}]
                    #print(key)
                    #print(str(summarize_result[key]))
                    
                    try:
                        srk_dict = json.loads(summarize_result[key])
                    except:
                        srk_dict = summarize_result[key]
                    
                    if type(srk_dict) == dict: #如果是流动相和分散相信息应该是字典
                        info_needed_print_in_this_phase = []
                        for phase_info_key in srk_dict: #对于每个相的每个标签, 若不为空则加入
                            #print(phase_info_key)
                            if srk_dict[phase_info_key] not in None_list:
                                info_needed_print_in_this_phase.append({phase_info_key:srk_dict[phase_info_key]})
                        if info_needed_print_in_this_phase != []: #这一相有信息需要打印
                            info_needed_print.append((key + ': ')) #把标题加进去,加上冒号
                            for i in info_needed_print_in_this_phase:
                                info_needed_print.append(i) #加入到大列表里面
                            #info_needed_print.append('\n') #用于分割不同的key
                    elif type(srk_dict) == str: #如果是液滴类型应该是字符串
                        if srk_dict not in None_list:
                            info_needed_print.append(key+': '+srk_dict) #直接加入键值对
                            #info_needed_print.append('\n') #用于分割不同的key
        except:
            error += 1
            print("格式不对",error,"//",total)
            
        if info_needed_print != []: #如果不为空
            #现在大列表已经做完
            f_out.write('**提取结果：**')
            f_out.write('\n\n') 
            f_out.write('~~~json\n') #代码块
            for i in info_needed_print:
                if type(i) == dict:
                    f_out.write(json.dumps(i))
                else:
                    f_out.write(i)
                f_out.write('\n')
            f_out.write('~~~\n') #代码块结束
                        
                
        
        f_out.write('\n\n\n')
    f_out.close()