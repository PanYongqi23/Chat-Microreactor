#!/usr/bin/env python
# coding: utf-8

from api import *

#读取向量
import numpy as np
embedding_array_2d = np.load("..//data//abstract_embeddings_2d.npy")
import pandas as pd
import sqlite3
from config import *

def search_row_by_value(conn, table_name, column_name, search_value):
    # 创建一个SQL查询语句来查找匹配的行
    query = f"SELECT * FROM {table_name} WHERE {column_name} = ?"
    
    # 使用pandas的read_sql_query方法执行查询
    df_result = pd.read_sql_query(query, conn, params=[search_value])
    
    # 返回查询结果
    return df_result

from sklearn.preprocessing import MinMaxScaler

def find_closest_normalized_rows_numpy(conn, table_name, search_criteria, top_n=10):
    # 从数据库中读取整个表
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    # 选择用于比较的列
    columns_to_normalize = list(search_criteria.keys())
    df_to_normalize = df[columns_to_normalize]
    
    # 归一化这些列
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_to_normalize), columns=columns_to_normalize, index=df.index)
    
    # 将搜索条件归一化
    search_values_normalized = scaler.transform([list(search_criteria.values())])[0]
    
    # 计算每一行与搜索条件的欧几里得距离
    df_normalized['distance'] = np.sqrt(((df_normalized - search_values_normalized) ** 2).sum(axis=1))
    
    # 根据距离排序，并选择距离最小的top_n行
    closest_rows_index = df_normalized.nsmallest(top_n, 'distance').index
    closest_rows = df.loc[closest_rows_index]
    
    return closest_rows

def cosine_similarity_vectorized(x, A):
    # 计算x的范数
    x_norm = np.linalg.norm(x)
    
    # 计算A每行的范数
    A_norms = np.linalg.norm(A, axis=1)
    
    # 计算x与A中每一行的点积
    dot_products = np.dot(A, x)
    
    # 计算余弦相似度
    cosine_similarities = dot_products / (x_norm * A_norms)
    
    return cosine_similarities

#给定文本出向量
# async def text_2_np(text):
def text_2_np(text):
    task = embedding_40(text)#embedding_async_40(text)
    result_lst = task #await task
    #print(result_lst)
    result_np = np.array(result_lst)
    return result_np

#给定文本输出表格
# async def text_related_df(text,conn,table_name, top_n=10):
def text_related_df(text,conn,table_name, top_n=3):
    global embedding_array_2d
    
    # 从数据库中读取整个表
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    embe = text_2_np(text) #await text_2_np(example_sentence)
    similarities = cosine_similarity_vectorized(embe, embedding_array_2d)
    #example_similarities.size


    top_indices = similarities.copy().argsort()[-top_n:][::-1]
    return df.loc[top_indices]

# 定义一个函数将DataFrame转换为Markdown格式的表格字符串
def df_to_markdown(df, bold_heading=False):
    # DataFrame的列名
    columns = df.columns.values
    # DataFrame的行数和列数
    nrow, ncol = df.shape
    
    # 开始构建Markdown表格
    markdown_str = "| " + " | ".join(columns) + " |\n"
    # 分隔表头和表格内容
    if bold_heading:
        markdown_str += "|---" * ncol + "|\n"
    else:
        markdown_str += "|---" * ncol + "|"
    
    # 填充表格内容
    for _, row in df.iterrows():
        markdown_str += "\n| " + " | ".join(str(cell).replace("\n", "<br>") for cell in row) + " |"
    
    return markdown_str

# ## 前端

import copy
def qaa(question,conn_x):
    
    global history_global

    conn = conn_x
    history = copy.deepcopy(history_global)
    
    related_df = text_related_df(question,conn,'basic_prop', top_n=3) #await text_related_df(example_sentence,conn,table_name, top_n=10)
    related_df_str = df_to_markdown(related_df)
    
    instruction = """
You are a chemistry engineering assistant that specifically handles questions and give advice related to microreactor design based on the papers you have reviewed.
Answer the question using the provided context. 
The table in the Context contains some information including the width of the channel(mm), the depth of the channel(mm), Q (flow rate, m^3/s), etc. 
Items prefixed with 'c_' indicate properties of the continuous phase, while those prefixed with 'd_' indicate properties of the dispersed phase. 
When responding to questions, if it involves designing a reactor, please provide a written description as well as the information on the width and depth of the channels (unit:mm) and Q (flow rate, m^3/s)(No other properties are required.), for both the continuous and dispersed phases, for my reference.
If the question is not very relevant to chemistry engineering, respond with 'Based on the information vailable from the paper I have read so far, I cannot provide a reliable answer to this question. Please revise your question.
\n\nContext:\n
    """
    messages = [{"role": "system", "content": instruction + related_df_str}]
    
    for his in history:
        messages.append(his)
        
    response = Chat(question,messages)
    
    answer = ''
    for chunk in response:
        if chunk != None:
            answer += chunk
            yield chunk
    
    history_global.append({"role": "user", "content": question})
    history_global.append({"role": "assistant", "content": answer})
    return

