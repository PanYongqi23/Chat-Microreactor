import os
import json
import pandas as pd
from api import *
from io import StringIO
import warnings
warnings.filterwarnings("ignore")
import math

input_path = 'summarized_result'
input_list = os.listdir(input_path)

output_path = 'summarized_result_change_unit'

columns = ['fluid composition', 'flow rate', 'density', 'viscosity', 'interface tension', 'channel’s width', 'channel’s depth', 'weber number', 'capillary number']

# Load prompts from a file
prompts_list = []
with open('prompts.txt','r',encoding = 'utf-8') as f_prompt:
    str_list = f_prompt.readlines()
    for i in str_list:
        prompt = i.split("\n")[0]
        prompts_list.append(prompt)

# Function to process a single input
async def convert_one_column(propmt_i, column): # column is a list
    prompt_combine = propmt_i + str(column)
    one_rst = await Completion_async_40(prompt_combine)
    return one_rst

# Function to filter results and convert to DataFrame
# Remove explanations if added by GPT
def filter_lines_with_symbol(text, symbol, symbol_ex = "--"):
    """
    Filter out lines that do not contain the specified symbol, and exclude lines with another specified symbol.

    Parameters:
    text (str): Input multiline text.
    symbol (str): Symbol to include.
    symbol_ex (str): Symbol to exclude.

    Returns:
    str: Filtered text with lines containing the specified symbol and not containing the excluded symbol.
    """
    # Split the text into lines
    lines = text.split('\n')
    
    # Filter out lines that do not contain the specified symbol and exclude lines with another specified symbol
    filtered_lines = [line for line in lines if symbol in line and symbol_ex not in line]
    
    # Reassemble the filtered lines into a string
    filtered_text = '\n'.join(filtered_lines)
    
    return filtered_text

def rst_2_df(rst_str): # Convert input to DataFrame, input is a string
    # Use StringIO to treat the string as a file object
    data = StringIO(rst_str)

    # Read data and convert to DataFrame
    df = pd.read_csv(data, delimiter='|', skipinitialspace=True)

    # Trim spaces from column names
    df.columns = df.columns.str.strip()

    # Remove completely empty columns
    df = df.dropna(axis=1, how='all')
    
    return df

import ast

def process_converted_result(cell):  # Function to process flow rate, others do not need processing
    try:
        if cell == "'N/A'":
            return 'N/A', 'N/A'
        elif isinstance(cell, str) and cell.startswith("["):
            q_values = []
            v_values = []
            for item in ast.literal_eval(cell):
                if 'q' in item:
                    q_values.append(item['q'])
                    v_values.append('N/A')
                elif 'v' in item:
                    v_values.append(item['v'])
                    q_values.append('N/A')
            return q_values, v_values
        else:
            cell_dict = ast.literal_eval(cell)
            if 'q' in cell_dict:
                return cell_dict['q'], 'N/A'
            elif 'v' in cell_dict:
                return 'N/A', cell_dict['v']
            else:
                return 'N/A', 'N/A'
    except (ValueError, SyntaxError) as e:
        print(f"Error processing cell '{cell}': {e}")
        return 'N/A', 'N/A'

# Check if strings in a list are in a given string
def check_strings_in_input(str_list, input_str):
    for string in str_list:
        if string in input_str:
            return True
    return False

def stand_type(unstand_str): # Standardize droplet types, input is a string, output is a standardized string
    type_dict = {
        "jetting":["Jet","jet"],
        "dripping":["drip","Drip"],
        "squeezing":["squeez","Squeez"]        
        }
    num_of_stand_word = 0
    stand_type = None
    for key in type_dict:
        check_result = check_strings_in_input(type_dict[key], unstand_str) 
        num_of_stand_word += check_result
        if check_result == True:
            stand_type = key
    if num_of_stand_word == 1: # Contains only one keyword
        return stand_type
    return 'N/A' # Otherwise, return 'N/A'

def stand_type_for_list_and_str(input_obj):
    """
    This function applies the 'stand_type' function to either a string or a list of strings.
    If the input is a string, it directly applies 'stand_type'.
    If the input is a list, it applies 'stand_type' to each element in the list.

    Args:
    - input_obj (str or list of str): The input string or list of strings.

    Returns:
    - str or list of str: The transformed string or list of strings.
    """

    # Check if the input is a string
    if isinstance(input_obj, str):
        return stand_type(input_obj)
    # Check if the input is a list
    elif isinstance(input_obj, list):
        return [stand_type(item) for item in input_obj]
    else:
        raise ValueError("Input must be a string or a list of strings.")

# Define a function to check if a cell is empty, math.nan, or contains 'N/A'
def check_cell(cell):
    if cell == '' or cell is math.nan or isinstance(cell, str) and 'N/A' in cell:
        return True
    return False

# Function to write log files
def write_log(df_ori,col_name): # First argument is df_rst, second is the column name for log writing
    print("Writing log")
    df_log = df_ori.iloc[:, 0:2].copy()

    # Use applymap to apply the check function to each cell, then use all(axis=1) to check if each row meets the condition
    rows_to_drop = df_log.applymap(check_cell).all(axis=1)

    # Delete rows that meet the condition
    df_log = df_log[~rows_to_drop]
    
    # Write to file, excluding row and column labels
    file_path = f"chang_unit_log//chang_unit_log_{col_name}.csv"
    with open(file_path, 'a',encoding = 'utf-8') as f:
        df_log.to_csv(f, index=False, header=False)
    print("Log written")

import csv

def write_time_log(log_list, file_name='change_unit_time_log.csv'):
    # Ensure the input is a list
    if not isinstance(log_list, list):
        raise ValueError("Input must be a list")
    
    # Write to a CSV file
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(log_list)
        
async def deal_with_one_file(file, k_this_task):  # Processing of a single file
    
    file_path = input_path + '//' + file
    
    title = file[:-5]
    
    output_file_path = output_path + '//' + title + '.csv'

    # Check if the file has already been generated
    if not os.path.exists(output_file_path):  # If it hasn't been generated
        
        f_in = open(file_path, 'r', encoding='utf-8')
        json_str_list = f_in.readlines()
        f_in.close()
        
        df_c = pd.DataFrame(columns=columns)  # Continuous phase
        df_d = pd.DataFrame(columns=columns)  # Dispersed phase
        df_t = pd.DataFrame(columns=['droplets type'])  # Droplet type
        
        # Each dimension of each document occupies one ask
        
        for json_str in json_str_list:
            # Slice each paragraph of text
            data_dict = json.loads(json_str[:-1])
                    
            if data_dict["if_useful"] == True:
                
                summarize_result = json.loads(data_dict["summarize_result"])
                try:
                    df_c = pd.concat([df_c, pd.DataFrame([summarize_result['continuous phase']])])
                    df_d = pd.concat([df_d, pd.DataFrame([summarize_result['dispersed phase']])])
                    df_t = pd.concat([df_t, pd.DataFrame([{'droplets type': summarize_result['droplets type']}])])
                except:
                    pass
                    
        # Now that we have 3 dataframes, the next step is to package each column and send it to chatgpt
        df_c.fillna('N/A', inplace=True)
        df_d.fillna('N/A', inplace=True)
        df_t.fillna('N/A', inplace=True)

        # Now send each column to GPT4 for unit conversion
        
        # First standardize df_t
        df_t[df_t.columns[0]] = df_t[df_t.columns[0]].apply(stand_type_for_list_and_str)
        
        # The first column is the composition, no need to convert
        
        df_total = df_t.copy().iloc[:, 0].to_frame(name='type')
        
        # The row indices are all 0, let's reset them
        df_total.reset_index(drop=True, inplace=True)
        df_dict = {"c": df_c, "d": df_d}
        
        # Information list that needs to be written to the log, write it to the log after the entire document is converted
        logs_list = []  # List objects are all dictionaries
        time_logs_list = []
        
        # Add continuous and dispersed phase information
        for key in df_dict:
        
            start_time = time.time()  # get timestamp

            df_dict[key].reset_index(drop=True, inplace=True)
            df_total[f'{key}_content'] = df_dict[key].copy().iloc[:, 0]
            # The second column is the flow rate. It can be divided into two cases, stored as {"v": velocity} and {"q": flow rate}
            print(f'Processing document {k_this_task} {key} phase part 2//9')
            str_rst = await convert_one_column(prompts_list[0], df_dict[key].iloc[:, 1].tolist())
            
            str_rst = filter_lines_with_symbol(str_rst, "|", "--")  # Format processing
            df_rst = rst_2_df(str_rst)  # Convert to dataframe
            #print(df_rst)
            
            # Generate log section for statistics
            logs_list.append({"df_rst": df_rst, "column": columns[1]})
            
            end_time = time.time()  # end time
            elapsed_time = end_time - start_time  # calculate elapsed time
            time_logs_list.append([k_this_task, key, 2, len(df_dict[key].iloc[:, 1].tolist()), elapsed_time, len(prompts_list[0]), len(str_rst)])
            
            df_rst[['Q', 'V']] = df_rst[df_rst.columns[1]].apply(lambda x: pd.Series(process_converted_result(x)))  # Separate velocity and flow rate
            df_total[f'{key}_Q'] = df_rst.copy()['Q']
            df_total[f'{key}_V'] = df_rst.copy()['V']
            # The third column is density, just process it normally
            # The fourth column is viscosity
            # The fifth column is surface tension
            # The sixth column is channel width.
            # The seventh column is channel height.
            # The eighth column is the Weber number, no unit conversion needed, just cleaning
            # The ninth column is the capillary number, no unit conversion needed, just cleaning
            # columns = ['fluid composition', 'flow rate', 'density', 'viscosity', 'interface tension', 'channel’s width', 'channel’s depth', 'weber number', 'capillary number']
            for j in range(3, 10):  # From the third to the 10th column
                
                start_time = time.time()  # get timestamp
                
                print(f'Processing document {k_this_task} {key} phase part {j}//9')
                str_rst = await convert_one_column(prompts_list[j-2], df_dict[key].iloc[:, j-1].tolist())
                print(str_rst)
                str_rst = filter_lines_with_symbol(str_rst, "|", "--")  # Format processing
                df_rst = rst_2_df(str_rst)  # Convert to dataframe
                
                logs_list.append({"df_rst": df_rst, "column": columns[j-1]})
                
                end_time = time.time()  # end time
                elapsed_time = end_time - start_time  # calculate elapsed time
                time_logs_list.append([k_this_task, key, j, len(df_dict[key].iloc[:, j-1].tolist()), elapsed_time, len(prompts_list[j-2]), len(str_rst)])

                df_total[f'{key}_{columns[j-1]}'] = df_rst.copy().iloc[:, -1]  # Add to df_total
        
        # Save as a table to view
        df_total.to_csv(output_file_path)
        # Write to log
        for log_dict in logs_list:
            write_log(log_dict["df_rst"], log_dict["column"])
        for time_log in time_logs_list:
            write_time_log(time_log)
        
        
kt = 0

loop = asyncio.get_event_loop()
task_list = []
part = 16
for file_name in input_list:
#for file_name in input_list[0:(part+1)*20]:
    task_list.append(loop.create_task(deal_with_one_file(file_name, kt)))
    kt += 1
loop.run_until_complete(asyncio.wait(task_list))
