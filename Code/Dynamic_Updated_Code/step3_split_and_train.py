import numpy as np
from sklearn.model_selection import train_test_split

# ------------- Load data -------------
X = np.load('/Users/nahidkhan/Local Drive/Research/Landmark_Data/X100.npy')   # (515,100,126)
y = np.load('/Users/nahidkhan/Local Drive/Research/Landmark_Data/y.npy')      # (515,)
label_names = np.load('/Users/nahidkhan/Local Drive/Research/Landmark_Data/label_names.npy', allow_pickle=True)

print("X:", X.shape, "y:", y.shape)
print("Classes order:", label_names)

# ------------- Split 70/15/15 -------------
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# Save splits (optional but useful)
np.save("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_train.npy", X_train)
np.save("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_train.npy", y_train)
np.save("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_val.npy", X_val)
np.save("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_val.npy", y_val)
np.save("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_test.npy", X_test)
np.save("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_test.npy", y_test)
