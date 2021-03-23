from pathlib import Path
from utils.io import json_load


class Config:
    @classmethod
    def load(cls, config_json):
        return cls(json_load(config_json))

    def __init__(self, d) -> None:
        for k, v in d.items():
            if k.endswith("_dir"):
                self.__setattr__(k, Path(v))
            elif isinstance(v, dict):
                self.__setattr__(k, Config(v))
            else:
                self.__setattr__(k, v)

    def __repr__(self) -> str:
        return (
            "{"
            + "\n".join(
                map(
                    lambda s: "  " + s,
                    "\n".join(f"{k}: {v}" for k, v in self.__dict__.items()).split("\n"),
                )
            )
            + "}"
        )

    def __str__(self) -> str:
        return (
            "{\n"
            + "\n".join(
                map(
                    lambda s: "  " + s,
                    "\n".join(f"{k}: {v}" for k, v in self.__dict__.items()).split("\n"),
                )
            )
            + "\n}"
        )

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__.get(key)
