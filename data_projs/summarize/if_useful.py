import json

def judge_if_useful(summarize_result_str):
    info_needed_print = []
    if_useful_y = False 

    try: 
        summarize_result = json.loads(summarize_result_str)
        
        for key in summarize_result:
            none_list = [None, "N/A", "", {}]
            
            try:
                srk_dict = json.loads(summarize_result[key])
            except:
                srk_dict = summarize_result[key]
            
            if type(srk_dict) == dict:
                info_needed_print_in_this_phase = []
                for phase_info_key in srk_dict:
                    if srk_dict[phase_info_key] not in none_list:
                        info_needed_print_in_this_phase.append({phase_info_key: srk_dict[phase_info_key]})
                if info_needed_print_in_this_phase != []:
                    info_needed_print.append(key + ': ')
                    for i in info_needed_print_in_this_phase:
                        info_needed_print.append(i)
            elif type(srk_dict) == str:
                if srk_dict not in none_list:
                    info_needed_print.append(key + ': ' + srk_dict)
    
    except:
        pass
    
    if info_needed_print != []:
        if_useful_y = True
    return if_useful_y

A_dict = {
    "continuous phase": {
        "fluid composition": "N/A", 
        "flow rate": "N/A", 
        "density": "N/A", 
        "viscosity": "N/A", 
        "interface tension": "N/A",
        "channel’s width": "N/A",
        "channel’s depth": "N/A",
        "weber number": "N/A",
        "capillary number": "N/A"
    },
    "dispersed phase": {
        "fluid composition": "N/A", 
        "flow rate": "dsdsd", 
        "density": "N/A", 
        "viscosity": "N/A", 
        "interface tension": "N/A",
        "channel’s width": "N/A",
        "channel’s depth": "N/A",
        "weber number": "N/A",
        "capillary number": "N/A"
    },
    "droplets type": "N/A"
}

if __name__ == '__main__':
    A_str = json.dumps(A_dict)
    print(judge_if_useful(A_str))
