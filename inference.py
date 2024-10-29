# PyTorch
import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
# For dislaying images
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
# Loading dataset
from datasets import Dataset, load_dataset
# Transformers
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
# Matrix operations
import numpy as np
import torch.nn as nn
# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore")


image = ... # Include your PIL image here
label = ... # Inlucde label (if you want, otherwise None)

# Example
image = [Image.fromarray(np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)).convert("RGB")]
label = [None]



def predictor(image, label = [None]):

    # Load processor
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)

    # Compile data
    data = (image, label)

    # Create dataset
    x_ds = Dataset.from_dict({'img': data[0], 'label': data[1]})

    # Normalization
    mu, sigma = processor.image_mean, processor.image_std #get default mu,sigma
    size = processor.size
    norm = Normalize(mean=mu, std=sigma) #normalize image pixels range to [-1,1]


    # Resize 3x512x512 to 3x224x224 -> convert to Pytorch tensor -> normalize
    _transf = Compose([
        Resize(size['height']),
        ToTensor(),
        norm
    ])

    # Apply transforms to PIL Image and store it to 'pixels' key
    def transf(arg):
        arg['pixels'] = [_transf(image.convert('RGB')) for image in arg['img']]
        return arg

    # Apply transforms
    x_ds.set_transform(transf)


    # Load model
    num_classes = 4
    model_name = "google/vit-base-patch16-224"
    model_path = "/Users/xaviermootoo/Documents/VScode/google-datathon2024/best-model"
    model = ViTForImageClassification.from_pretrained(model_path, num_labels=4, ignore_mismatched_sizes=True)
    model.eval()  # Set model to evaluation mode



    # Single example for prediction (make sure 'transf' is applied correctly)
    example = x_ds[0]
    input = processor(images=example['img'], return_tensors="pt")  # Create input dictionary
    with torch.no_grad():
        outputs = model(**input)  # Pass the dictionary to the model
        logits = outputs.logits
        prediction = nn.Softmax(dim=1)(logits)  # Softmax to get probabilities

    # To get the predicted class
    predicted_class_index = logits.argmax(dim=1).item()
    print(f"Predicted Class Index: {predicted_class_index}")


if __name__ == "__main__":
    predictor(image)
