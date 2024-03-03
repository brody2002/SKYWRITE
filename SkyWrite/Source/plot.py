#NOTE: pip install the requirements.txt as well as these other imports if not installed



#Imports
import math
import torch
import pandas as pd
import csv
import vec2text
import openai
from openai import OpenAI
from openai.resources import embeddings as OpenAI_embed_class
import torch
import logging





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




# Specify the path to your CSV file
csv_file_path = 'TENSOR_EMBEDDINGS.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
# Convert the DataFrame to a numpy array
numpy_array = df.values
# Convert the numpy array to a PyTorch tensor
tensor_list_vectors = torch.tensor(numpy_array, dtype=torch.float)

print(tensor_list_vectors)





"USER INPUT SECTION: "
import numpy as np

from sklearn.decomposition import PCA

client = OpenAI(api_key="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")
embed_Class = OpenAI_embed_class.Embeddings(client)


def Add_Embedding(text_list, model="text-embedding-ada-002") -> torch.Tensor:


    batches = math.ceil(len(text_list) / 128)
    outputs = []
    for batch in range(batches):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]
        response = embed_Class.create(
            input=text_list_batch,

            model=model,
            encoding_format="float",  # override default base64 encoding...
        )
    outputs.extend([e.embedding for e in response.data])
    return torch.tensor(outputs) 

"[[1...2, 43, ]]"
def Input_Embedding(input_string :str, input_pca: PCA, tensor_list_vectors) -> torch.tensor:
    input_strings.append(input_string)


    #concat the tensors first and then run pca transform on it: 
    concatenated_list = torch.cat((tensor_list_vectors, Add_Embedding([input_string])), dim=0)
    # print("100 index: ",concatenated_list[100])
    # print("original length: ",len(concatenated_list), "length of 0 index: ", len(concatenated_list[0]))
    output = input_pca.transform(concatenated_list) # -> 2d embedding list

    return output, concatenated_list # ->tensorflow list


#Don't add user input into original matrix space. 
# Instead, have two separate matrix spaces and pca.transform(the second)







# Instantiate PCA
cur_pca = PCA(n_components=2)
# Fit PCA on your data and transform it

embeddings_2d = cur_pca.fit_transform(tensor_list_vectors)
# print("original length: ",len(embeddings_2d))


# embeddings_2d, tensor_list_vectors = Input_Embedding("Hello, I'm writing to test if the function PCA_function is working ",input_pca=cur_pca, tensor_list_vectors=tensor_list_vectors) #attempt
# embeddings_2d, tensor_list_vectors = Input_Embedding("Listening to music makes me feel calm.",input_pca=cur_pca, tensor_list_vectors=tensor_list_vectors) #attempt







import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Assuming the PCA step has been completed and embeddings_2d is available

# Create a scatter plot for all points
fig, ax = plt.subplots(figsize=(10, 6))

print("embeddings2d: ",len(embeddings_2d))
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







