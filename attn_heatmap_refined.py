from torch import nn
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from PIL import Image
import cv2

# GET ATTENTION AND INPUTS
attentions = outputs.attentions
inputs = #{INPUT GENERATED FROM PROCESSOR FUNCTION} - ex) processor(images=exi, return_tensors="pt")

# FUNCTIONS
def visualize_attention(inputs, attentions):
    patch_size = 16

    w_featmap = inputs.pixel_values.shape[-2] // patch_size
    h_featmap = inputs.pixel_values.shape[-1] // patch_size


    nh = attentions.shape[1]  # number of head

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()

    return attentions


def plot_attention(attentions):
    n_heads = attentions.shape[0]
    for i in range(n_heads):
            plt.subplot(ceil(n_heads/3), 3, i+1)
            plt.imshow(attentions[i], cmap='inferno')
            plt.title(f"Head n: {i+1}")

    plt.tight_layout()
    plt.show()

def normalize_array(arr, lower_bound, upper_bound):
    # Find the minimum and maximum values in the array
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Scale the array linearly between lower_bound and upper_bound
    normalized_arr = lower_bound + (upper_bound - lower_bound) * (arr - min_val) / (max_val - min_val)
    
    return normalized_arr



# MAIN CODE

# GET ATTENTION MATRIX
n = visualize_attention(inputs, attentions[-1])

# TAKE MEAN OF ATTENTION HEADS
n = np.mean(n, 0, keepdims=True)

# PLOT ATTENTION MATRIX
plot_attention(n)

# REMOVE FIRST DIMENSION OF ATTENTIONS
n = n[0]
normalized = normalize_array(n, 0, 255)

# CREATE black and white heatmap
heatmap_image = Image.fromarray((normalized).astype(np.uint8))
heatmap_image = heatmap_image.convert("RGB")


# CREATE RGB HOT HEATMAP
open_cv_image = np.array(heatmap_image)
colormap = plt.get_cmap('inferno')
heatmap = (colormap(open_cv_image)).astype(np.uint8)[:,:,:3]
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
heatmap = cv2.applyColorMap(open_cv_image, cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

