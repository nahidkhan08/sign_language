import numpy as np
y = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y.npy")
label_names = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/label_names.npy", allow_pickle=True)

for i, name in enumerate(label_names):
    print(name, (y==i).sum())
