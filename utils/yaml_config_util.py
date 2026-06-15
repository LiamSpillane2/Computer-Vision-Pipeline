import yaml
import os
from pathlib import Path

def load_yaml(file_path:str, train_path:str, val_path:str, test_path:str, names:list) -> str:

    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        file_path (str): The destination path decided for the YAML configuration file.
        train_path (str): The path to the training data.
        val_path (str): The path to the validation data.
        test_path (str): The path to the test data.
        names (list): A list of class names.
        
    """
    #Root path for the YAML file.
    root = Path(__file__).resolve().parents[1]
    
    #Number of classes.
    nc = len(names) 
    
    #Configuring the data content for the YAML file.
    data_content = {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': nc,
        'names': [i for i in names]
    }
    
    #Writing the data content to the YAML file.
    file_name = "data.yaml"
    os.makedirs(os.path.dirname(f'{root}/{file_path}/{file_name}'), exist_ok=True)
    with open(f'{root}/{file_path}/{file_name}', 'w+') as yaml_file:
        yaml.dump(data_content, yaml_file, default_flow_style=False)
    file_path = os.path.join(root, file_path, file_name)
    
    print(f"YAML configuration file '{file_name}' has been created successfully at '{file_path}'.")
    
    return file_path

