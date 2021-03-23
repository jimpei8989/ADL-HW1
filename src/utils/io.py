import json
import pickle


def json_load(file):
    with open(file, "r") as f:
        return json.load(f)


def json_dump(obj, file):
    with open(file, "w") as f:
        return json.dump(obj, f)


def pickle_load(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def pickle_dump(obj, file):
    with open(file, "wb") as f:
        return pickle.dump(obj, f)
