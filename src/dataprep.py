import torch
from torchvision import datasets, transforms
import cv2
import numpy as np

# Define transformation: convert PIL image to tensor and normalize to [0,1]
transform = transforms.ToTensor()

# Load MNIST digits
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Load EMNIST letters (we will filter for x and y)
emnist_train = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
emnist_test  = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

# Filter EMNIST to only include classes for 'x' and 'y'.
# According to EMNIST Letters mapping: if A=1, B=2, ..., X=24, Y=25, Z=26 (1-indexed in some sources),
# then 'X' and 'Y' would correspond to labels 24 and 25. However, to confirm, we'll adjust if needed.
# Torchvision EMNIST might label A=0,... Z=25 (0-indexed). In that case, x=23, y=24 (0-indexed).
# We'll check both possibilities and filter accordingly.
x_indices = []
y_indices = []
for idx, (_, label) in enumerate(emnist_train):
    # If label 23 or 24 appear, collect indices. (We assume 0-indexed here for safety.)
    if label == 23:  # possibly 'x'
        x_indices.append(idx)
    elif label == 24:  # possibly 'y'
        y_indices.append(idx)
# If we didn't get any, adjust assuming 1-indexed labels in dataset (rare in Torch, but just in case).
if len(x_indices) == 0:
    for idx, (_, label) in enumerate(emnist_train):
        if label == 23+1:  # 24 if 'X' was 24 (1-indexed)
            x_indices.append(idx)
        elif label == 24+1:  # 25
            y_indices.append(idx)

# Subset the EMNIST dataset for training and testing to only include those indices
# We'll create new datasets by collecting the filtered data.
emnist_x_train_data = [emnist_train[i][0] for i in x_indices]  # images for 'x'
emnist_y_train_data = [emnist_train[i][0] for i in y_indices]  # images for 'y'
emnist_x_train_labels = [10]*len(x_indices)  # label 10 for 'x'
emnist_y_train_labels = [11]*len(y_indices)  # label 11 for 'y'

# Do the same for EMNIST test set
x_indices_test = [idx for idx, (_, label) in enumerate(emnist_test) if label in (23, 24, 24+1, 25)]
# (We'll reuse logic; in practice ensure it matches the train label scheme)
emnist_x_test_data = []
emnist_y_test_data = []
emnist_x_test_labels = []
emnist_y_test_labels = []
for idx, (img, label) in enumerate(emnist_test):
    if label == 23 or label == 24+1:  # 'x'
        emnist_x_test_data.append(img)
        emnist_x_test_labels.append(10)
    elif label == 24 or label == 25+1:  # 'y'
        emnist_y_test_data.append(img)
        emnist_y_test_labels.append(11)

# Prepare MNIST data and labels (digits 0-9 keep their labels 0-9)
mnist_train_data = [img for (img, _) in mnist_train]
mnist_train_labels = [label for (_, label) in mnist_train]
mnist_test_data  = [img for (img, _) in mnist_test]
mnist_test_labels  = [label for (_, label) in mnist_test]

# Generate synthetic symbols data for training:
symbols = ['+', '-', '×', '÷', '=', '^']
symbol_label_map = {'+':12, '-':13, '×':14, '÷':15, '=':16, '^':17}

def generate_symbol_image(sym):
    """Generate a synthetic 28x28 image for the given symbol using OpenCV drawing."""
    # Start with a white image
    img = np.full((28,28), 255, dtype=np.uint8)
    if sym == '+':
        # Draw a plus: two perpendicular lines
        cv2.line(img, (14, 5), (14, 23), 0, 3)       # vertical line
        cv2.line(img, (5, 14), (23, 14), 0, 3)       # horizontal line
    elif sym == '-':
        cv2.line(img, (5, 14), (23, 14), 0, 3)       # horizontal line (minus)
    elif sym == '×':
        # Draw an 'x' for multiplication (two diagonal lines crossing)
        cv2.line(img, (5, 5), (23, 23), 0, 3)
        cv2.line(img, (5, 23), (23, 5), 0, 3)
    elif sym == '÷':
        # Draw division sign: a horizontal line with a dot above and below
        cv2.line(img, (5, 14), (23, 14), 0, 3)
        cv2.circle(img, (14, 7), 2, 0, -1)  # top dot
        cv2.circle(img, (14, 21), 2, 0, -1) # bottom dot
    elif sym == '=':
        # Draw equals: two horizontal lines
        cv2.line(img, (5, 11), (23, 11), 0, 3)
        cv2.line(img, (5, 17), (23, 17), 0, 3)
    elif sym == '^':
        # Draw caret: a simple triangle shape (^) 
        # We'll draw it as two diagonal lines meeting at a point
        cv2.line(img, (10, 18), (14, 10), 0, 3)
        cv2.line(img, (14, 10), (18, 18), 0, 3)
    return img

# Generate multiple samples per symbol for training (to simulate a dataset)
symbol_train_data = []
symbol_train_labels = []
for sym in symbols:
    for _ in range(1000):  # generate 1000 samples for each symbol
        img = generate_symbol_image(sym)
        # Optionally, we could add slight random noise or shifts here for augmentation
        # Convert numpy array to torch tensor
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)  # shape (1,28,28)
        symbol_train_data.append(img_tensor)
        symbol_train_labels.append(symbol_label_map[sym])

# For testing, generate a smaller set
symbol_test_data = []
symbol_test_labels = []
for sym in symbols:
    for _ in range(200):  # 200 samples each for test
        img = generate_symbol_image(sym)
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        symbol_test_data.append(img_tensor)
        symbol_test_labels.append(symbol_label_map[sym])

# Combine all training data and labels
train_images = mnist_train_data + emnist_x_train_data + emnist_y_train_data + symbol_train_data
train_labels = mnist_train_labels + emnist_x_train_labels + emnist_y_train_labels + symbol_train_labels

# Combine all test data and labels
test_images = mnist_test_data + emnist_x_test_data + emnist_y_test_data + symbol_test_data
test_labels = mnist_test_labels + emnist_x_test_labels + emnist_y_test_labels + symbol_test_labels

# Convert lists to tensors/datasets for PyTorch
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels  = torch.tensor(test_labels, dtype=torch.long)
# Stack image tensors into one tensor of shape (N, 1, 28, 28)
train_images_tensor = torch.stack([img if isinstance(img, torch.Tensor) else img[0] for img in train_images])
test_images_tensor  = torch.stack([img if isinstance(img, torch.Tensor) else img[0] for img in test_images])

# Create TensorDataset for convenient loading
train_dataset = torch.utils.data.TensorDataset(train_images_tensor, train_labels)
test_dataset  = torch.utils.data.TensorDataset(test_images_tensor, test_labels)
print(f"Training set size: {len(train_dataset)} images, Test set size: {len(test_dataset)} images.")
