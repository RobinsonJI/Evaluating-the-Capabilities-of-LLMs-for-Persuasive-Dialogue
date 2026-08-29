import pandas as pd
from pathlib import Path

def load_testset_prompts(file_name, sep="\t", encoding="utf-16"):

    experiments_dir = Path(__file__).parent.parent / "data" / "experiments" 

    experiments_file = experiments_dir / file_name

    try:
        suffix = experiments_file.suffix.lower()

        if suffix == '.csv':
            df = pd.read_csv(experiments_file, sep=sep, encoding=encoding)
        elif suffix == '.jsonl':
            df = pd.read_json(experiments_file, lines=True, sep=sep, encoding=encoding)
        elif suffix == '.json':
            df = pd.read_json(experiments_file, sep=sep, encoding=encoding)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    except:
        raise FileNotFoundError(f"Experiment file not found: '{experiments_file}'")

    return df

def create_experiments_df(test_df, model_config):

    if "repetitions" in model_config:
        repetitions = model_config["repetitions"]
    else:
        repetitions = 1

    if "temperature" in model_config:
        temperature = model_config["temperature"]
    else:
        temperature = 0

    if "top_p" in model_config:
        top_p = model_config["top_p"]
    else:
        top_p = 0

    if "seed" in model_config:
        seed = model_config["seed"]
    else:
        seed = 0

    experiments_data = []
    for index, row in test_df.iterrows():
        for model in model_config["models"]:
            for repetition in range(0, repetitions):
                sample = {
                    "testset_row_index" : index,
                    "repetition" : repetition,
                    "dialogue_turn_1" : row["dialogue_turn_1"],
                    "dialogue_turn_2" : row["dialogue_turn_2"],
                    "target" : row["target"],
                    "label" : row["label"],
                    "predictor" : model,
                    "speaker" : row["speaker"],
                    "temp" : temperature,
                    "top_p" : top_p,
                    "seed" : seed
                    }
                experiments_data.append(sample)

    df = pd.DataFrame.from_dict(experiments_data)

    save_dir_path = Path(__file__).parent.parent / "data" / "experiments" / "metadata" 

    if not save_dir_path.exists():
        save_dir_path.resolve().mkdir(parents=True, exist_ok=False)

    
    save_file_path = save_dir_path / "experiments.csv"
    
    df.to_csv(save_file_path, index=False, encoding='utf-16', sep="\t")

    experiments_list = []
    for index, row in df.iterrows():
        row_data = row.to_dict()
        experiments_list.append(row_data)

    return experiments_list