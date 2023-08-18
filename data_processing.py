import pandas
from datasets import load_dataset

def load_data_from_json(json_file_path):
    return load_dataset("json", data_files=json_file_path)