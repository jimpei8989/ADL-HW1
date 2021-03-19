from tqdm import tqdm


def tqdmm(iterable, desc=""):
    return tqdm(iterable, desc=desc, ncols=80, leave=True)
