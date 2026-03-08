# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Include the problem statement and Dataset
</br>
To find the defective or not defective capaciter from a given image, the given image is off size 224 X 224.

### Dataset:

<img width="584" height="264" alt="image" src="https://github.com/user-attachments/assets/25013923-2f2a-447b-b0f9-7fa444e1967f" />


## DESIGN STEPS
### STEP 1:
Import the necessary modules and libraries

### STEP 2:
Load the dataset, after unzipping the data zip file

### STEP 3:
Use a dataloader and get a trian_loader and test_loaders


### STEP 4:
Load the pre-trained model and change the feature of the output linear layer

### STEP 5:
Instantiate the BCEWithLogitsLoss loss function and optimizer

### STEP 6:
Train the pre-trained model.

### STEP 7:
Evaulate the model with test data and create the iteration loss, validation loss graph

### STEP 8:
Create a Confustion matrix and classification report

### STEP 9:
Give a custom input and predict the output

## PROGRAM
<br/>

```python
# Load Pretrained Model and Modify for Transfer Learning
from torchvision.models import VGG19_Weights
model_subhash = models.vgg19(weights=VGG19_Weights.DEFAULT)


# Modify the final fully connected layer to match the dataset classes

num_classes = len(train_dataset.classes)
in_features = model_subhash.classifier[-1].in_features
model_subhash.classifier[-1] = nn.Linear(in_features, 1)


# Include the Loss function and optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_subhash.parameters(),lr=0.01)



# Train the model

## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:        ")
    print("Register Number:        ")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
</br>

<img width="757" height="800" alt="image" src="https://github.com/user-attachments/assets/8674b86b-70f9-4acf-af20-13c4e41477a5" />


### Confusion Matrix

</br>

<img width="660" height="601" alt="image" src="https://github.com/user-attachments/assets/f2390f70-aa13-47c3-aba6-7dce3b6ed541" />


### Classification Report

</br>

<img width="477" height="198" alt="image" src="https://github.com/user-attachments/assets/5bb748f8-f020-47fd-9179-3566336a3778" />


### New Sample Prediction
</br>

<img width="355" height="399" alt="image" src="https://github.com/user-attachments/assets/e4286338-cdd0-4404-9e88-910f24ac9af8" />

<img width="335" height="384" alt="image" src="https://github.com/user-attachments/assets/b1233417-62d2-4b96-b938-06fb2a9e4f05" />

## RESULT
Thus, a transfer learner program is implemented for classification using VGG-19 architecture.
