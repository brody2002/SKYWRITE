#Imports
import math
import torch
import pandas as pd
import csv


# Specify the path to your CSV file
csv_file_path = 'OUTPUT_STRINGS.csv'

# Initialize an empty list to store the strings
input_strings = []

# Open the CSV file and read its contents
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        input_strings.append(row[0])
input_strings.pop(0)



import pandas as pd

# Specify the path to your CSV file
csv_file_path = 'TENSOR_EMBEDDINGS.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

import torch

# Convert the DataFrame to a numpy array
numpy_array = df.values

# Convert the numpy array to a PyTorch tensor
tensor_list_vectors = torch.tensor(numpy_array, dtype=torch.float)






import numpy as np
from sklearn.decomposition import PCA

# Instantiate PCA
pca = PCA(n_components=2)
# Fit PCA on your data and transform it
embeddings_2d = pca.fit_transform(tensor_list_vectors)











import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Assuming the PCA step has been completed and embeddings_2d is available

# Create a scatter plot for all points
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

# Annotation for displaying the coordinates, initially hidden
annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    # Get the index of the hovered point
    index = ind["ind"][0]  # Assuming you want the first if there are multiple
    # Get the position of the hovered point
    pos = scatter.get_offsets()[index]
    annot.xy = pos
    # Format and display the annotation text (Index and PCA coordinates)
    text = f"Index: {index}, Coordinates: {pos}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    # Check if the event is over the scatter plot area
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

# Connect the hover event
fig.canvas.mpl_connect("motion_notify_event", hover)

plt.title('PCA Reduced Embeddings with Hover Information')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
