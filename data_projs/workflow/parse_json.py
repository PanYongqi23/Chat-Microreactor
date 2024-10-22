import os
import ast
import torch

Raw_PATH = r".\data\temp_results"
Processed_PATH = r".\data\processed"

# Create the directories if they do not exist
for path in [Raw_PATH, Processed_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# parse_json is used to load data from json file, and save them as .pth file for faster loading
def parse_json(json_path):
    f = open(json_path, "r", encoding="utf-8")
    json_str_list = f.readlines()  # 所有的json
    f.close()
    data_dict = {"labels": [], "embeddings": []}
    for line in json_str_list:
        line = line.replace(" ", "")
        if (
            line != None and line != "" and line != " " and line != "\n"
        ):  # 有些空行只有换行符
            if_useful_start = line.find(":") + 1
            if_useful_end = line.find(',"')
            if_useful = line[if_useful_start:if_useful_end]
            embedding_start = line.find(":", if_useful_end) + 1
            embedding_end = line.find(',"', embedding_start)
            embedding_str = line[embedding_start:embedding_end]
            embedding = ast.literal_eval(embedding_str)
            if if_useful == "false":
                label = False
            elif if_useful == "true":
                label = True
            data_dict["labels"].append(label)
            data_dict["embeddings"].append(embedding)

    data_dict["labels"] = torch.tensor(data_dict["labels"])
    #print(data_dict["embeddings"])
    data_dict["embeddings"] = torch.tensor(data_dict["embeddings"]).float()
    return data_dict

def parse_path(Raw_PATH):
    for sub in os.listdir(Raw_PATH):
        print(f'Parsing {sub}...')
        os.makedirs(Processed_PATH,exist_ok=True)
        #path = os.path.join(Raw_PATH, sub, "embedding_and_if_useful.json")
        path = os.path.join(Raw_PATH, sub)
        save_path = os.path.join(Processed_PATH, f"{sub}_data.pth")
        data = parse_json(path)
        torch.save(data, save_path)
        return data#当只有一个文件时

