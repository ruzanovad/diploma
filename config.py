from pathlib import Path

# This module offers classes representing filesystem paths
# with semantics appropriate for different operating systems.
# Path classes are divided between pure paths, which provide
# purely computational operations without I/O, and concrete paths,
# which inherit from pure paths but also provide I/O operations.


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq": 128,
        "d_model": 256,
        "datasource": "Helsinki-NLP/opus_books",
        "lang_src": "en",
        "lang_tgt": "ru",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / config["model_folder"] / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    # model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(config["model_folder"]).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
