import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("sachinpatel21/az-handwritten-alphabets-in-csv-format")
data_csv = pd.read_csv(f"{path}/handWritten.csv")
data = data_csv.to_numpy()

Y_train = data[:, 0]
X_train = data[:, 1:] / 255.0
print(Y_train)
print(X_train)
print(Y_train.shape)
print(X_train.shape)

image = X_train[4].reshape(28, 28)
plt.imshow(image, cmap="gray")
plt.title(f"Label: {Y_train[0]}")
plt.show()