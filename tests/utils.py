import numpy as np 

def get_indptr(max, batch):
    t = np.sort(np.random.choice(range(max), size=batch, replace=False)).astype("int32")
    return np.append(t,[max]).astype("int32")