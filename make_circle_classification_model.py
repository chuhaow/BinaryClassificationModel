import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

n_samples = 1000

#Create circles
FEATURES,labels = make_circles(n_samples=n_samples, noise=0.03,random_state=42)


# Turn data to tensors
FEATURES = torch.from_numpy(FEATURES).type(torch.float)
labels = torch.from_numpy(labels).type(torch.float)

# split data

FEAT_train, FEAT_test, lab_train, lab_test = train_test_split(FEATURES, labels, test_size=0.2, random_state=42)

# build Model
device = "cuda" if torch.cuda.is_available() else "cpu"

class CircleClassificationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers_1 = nn.Linear(in_features=2, out_features=16)
        self.layers_2 = nn.Linear(in_features=16, out_features=16)
        self.layers_3 = nn.Linear(in_features=16, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layers_3(self.relu(self.layers_2(self.relu(self.layers_1(x)))))
    
model_0 = CircleClassificationModel().to(device)

# Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc



# turn logits to prediction labels
y_pred_labels = torch.round(torch.sigmoid(model_0(FEAT_test.to(device))[:5]))
print(y_pred_labels)

# Train model
torch.manual_seed(42)
torch.cuda.manual_seed(42)
FEAT_train, FEAT_test, lab_train, lab_test  = FEAT_train.to(device), FEAT_test.to(device), lab_train.to(device), lab_test.to(device)

epoch_count = []
loss_values = []
test_loss_values = []
acc_values = []

epochs = 3000
for epoch in range(epochs):
    model_0.train()
    #forward
    label_logits = model_0.forward(FEAT_train).squeeze()
    label_pred = torch.round(torch.sigmoid(label_logits))

    # loss
    loss = loss_fn(label_logits,lab_train) #BCEWithLogitsLoss expects raw logits
    acc = accuracy_fn(y_true=lab_train, y_pred=label_pred)

    #backward
    optimizer.zero_grad()
    loss.backward()

    #grad desc
    optimizer.step()

    #test
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0.forward(FEAT_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits ,lab_test)
        test_acc = accuracy_fn(y_true=lab_test, y_pred=test_pred)
    
    if epoch % 20 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        acc_values.append(acc)
        print(f"Epoch:{epoch} | Loss: {loss} | Test loss: {test_loss} | Acc: {acc} | Test Acc: {test_acc}")

# Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

# Create model save path
MODEL_NAME = "make_circle_classification_model.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

# Save state dict
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)


# Visualize data 
plt.scatter(x=FEATURES[:,0], y=FEATURES[:,1], c=labels, cmap=plt.cm.RdYlBu)
plt.show()

