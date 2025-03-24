# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Develop a Convolutional Neural Network (CNN) to classify images into predefined categories. The dataset consists of labeled images split into training and testing sets, with preprocessing steps like resizing, normalization, and augmentation to enhance model performance. The trained model will be tested on new images to verify accuracy.

## Neural Network Model

![image](https://github.com/user-attachments/assets/cb131631-9bba-4dc8-a3c8-dd7a9b3c98ba)

## DESIGN STEPS

### STEP 1:
Import the necessary libraries such as NumPy, Matplotlib, and PyTorch.

### STEP 2:
Load and preprocess the dataset:

- Resize images to a fixed size (128×128).
- Normalize pixel values to a range between 0 and 1.
- Convert labels into numerical format if necessary.

### STEP 3:
Define the CNN Architecture, which includes:

- Input Layer: Shape (8,128,128)
- Convolutional Layer 1: 8 filters, kernel size (16×16), ReLU activation
- Max-Pooling Layer 1: Pool size (2×2)
- Convolutional Layer 2: 24 filters, kernel size (8×8), ReLU activation
- Max-Pooling Layer 2: Pool size (2×2)
- Fully Connected (Dense) Layer:
   - First Dense Layer with 256 neurons
   - Second Dense Layer with 128 neurons
   - Output Layer for classification
     
### STEP 4:
Define the loss function (e.g., Cross-Entropy Loss for classification) and optimizer (e.g., Adam or SGD).

### STEP 5:
Train the model by passing training data through the network, calculating the loss, and updating the weights using backpropagation.

### STEP 6:
Evaluate the trained model on the test dataset using accuracy, confusion matrix, and other performance metrics.

### STEP 7:
Make predictions on new images and analyze the results.

## PROGRAM

### Name: SARGURU K
### Register Number: 212222230134

```python
class CNNClassifier(nn.Module):
    def __init__(self): 
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x): 
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

```python
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)
```

```python
def train_model(model, train_loader, optimizer, criterion, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
```

## OUTPUT
### Training Loss per Epoch

![image](https://github.com/user-attachments/assets/8ac8888f-000b-420c-a08c-77d9a35e5a24)

### Confusion Matrix

![image](https://github.com/user-attachments/assets/63dbb982-8f10-4465-bf98-633b418f994a)

### Classification Report

![image](https://github.com/user-attachments/assets/23ff2123-4135-4d1b-9f7d-49cce21b98c8)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/6b450760-ea3e-4e4d-9b69-1018a65ccd01)


## RESULT

The Convolutional Neural Network (CNN) was successfully implemented for image classification. The model was trained on the dataset, and its performance was evaluated using accuracy metrics, confusion matrix, and classification report. Predictions on new sample images were verified, confirming the model's effectiveness in classifying images.
