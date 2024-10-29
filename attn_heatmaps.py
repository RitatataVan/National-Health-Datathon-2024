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
# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore")

def load_dataset(n,m):
    # Replace with your loading logic for CT images
    # Provide 'train', 'validation', and 'test' splits
    images_train = [Image.fromarray(np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)).convert("RGB") for _ in range(n)]
    labels_train = [np.random.randint(0, 4) for _ in range(n)]
    images_val = [Image.fromarray(np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)).convert("RGB") for _ in range(m)]
    labels_val = [np.random.randint(0, 4) for _ in range(m)]
    images_test = [Image.fromarray(np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)).convert("RGB") for _ in range(m)]
    labels_test = [np.random.randint(0, 4) for _ in range(m)]

    return (images_train, labels_train), (images_val, labels_val), (images_test, labels_test)


# Load dataset
train_data, val_data, test_data = load_dataset(12, 4)
trainds = Dataset.from_dict({'img': train_data[0], 'label': train_data[1]})
valds = Dataset.from_dict({'img': val_data[0], 'label': val_data[1]})
testds = Dataset.from_dict({'img': test_data[0], 'label': test_data[1]})


# Dictionaries for label mapping
# trainds.features['label'].names = ["Normal", "Acute", "Chronic", "Both"]
# valds.features['label'].names = ["Normal", "Acute", "Chronic", "Both"]
# testds.features['label'].names = ["Normal", "Acute", "Chronic", "Both"]

# itos = dict((k,v) for k,v in enumerate(trainds.features['label'].names))
# stoi = dict((v,k) for k,v in enumerate(trainds.features['label'].names))


# Load processor
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)


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
trainds.set_transform(transf)
valds.set_transform(transf)
testds.set_transform(transf)


# Display an image
idx = 0
ex = trainds[idx]['pixels']
ex = (ex+1)/2 #imshow requires image pixels to be in the range [0,1]
exi = ToPILImage()(ex)
plt.imshow(exi)
plt.show()


# Load model
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True)
print(model.classifier)


# Training arguments
args = TrainingArguments(
    f"random_test",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='logs',
    remove_unused_columns=False,
)

# Collate function
def collate_fn(examples):
    pixels = torch.stack([example["pixels"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixels, "labels": labels}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

# Create trainer
trainer = Trainer(
    model,
    args,
    train_dataset=trainds,
    eval_dataset=valds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

# Train
trainer.train()


# Evaluate on test set
outputs = trainer.predict(testds)
print(outputs.metrics)


inputs = processor(images=exi, return_tensors="pt")

outputs = model(**inputs, output_attentions=True)
logits = outputs.logits
attentions = outputs.attentions

print(logits)


from torch import nn

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


n = visualize_attention(inputs, attentions[-1])
n = np.mean(n, 0, keepdims=True)

import matplotlib.pyplot as plt
from math import ceil


def plot_attention(attentions):
    n_heads = attentions.shape[0]
    for i in range(n_heads):
            plt.subplot(ceil(n_heads/3), 3, i+1)
            plt.imshow(attentions[i], cmap='inferno')
            plt.title(f"Head n: {i+1}")

    plt.tight_layout()
    plt.show()

plot_attention(n)


display(exi)


print(n.shape)

n = n[0]
print(n.shape)

def normalize_array(arr, lower_bound, upper_bound):
    # Find the minimum and maximum values in the array
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Scale the array linearly between lower_bound and upper_bound
    normalized_arr = lower_bound + (upper_bound - lower_bound) * (arr - min_val) / (max_val - min_val)
    
    return normalized_arr

normalized = normalize_array(n, 0, 255)

heatmap_image = Image.fromarray((normalized).astype(np.uint8))
heatmap_image = heatmap_image.convert("RGB")
display(heatmap_image)


alpha = 0.3  # Adjust alpha as needed for transparency
blended_image = Image.blend(exi, heatmap_image, alpha)
display(blended_image)


print(heatmap_image)

open_cv_image = np.array(heatmap_image)

import cv2

colormap = plt.get_cmap('inferno')
heatmap = (colormap(open_cv_image)).astype(np.uint8)[:,:,:3]
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

heatmap = cv2.applyColorMap(open_cv_image, cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

plt.imshow(heatmap)

plt.imshow(open_cv_image)

# exil is PIL

exi_image = np.array(exi)
# exi_image = exi_image[:, :, ::-1].copy() # this is to convert RGB to BGR, but we want to stay RGB


plt.imshow(exi_image)


super_imposed_img = cv2.addWeighted(heatmap, 0.5, exi_image, 0.5, 0)

plt.imshow(super_imposed_img)