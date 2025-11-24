import numpy as np, hashlib

X = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X100.npy")

def hash_seq(seq):
    return hashlib.md5(seq.tobytes()).hexdigest()

hashes = [hash_seq(X[i]) for i in range(len(X))]
dup = len(hashes) - len(set(hashes))
print("Exact duplicate sequences:", dup)
