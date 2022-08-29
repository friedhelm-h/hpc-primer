import torch
import torch.nn as nn
from torch.utils.data import dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
from torch.utils.tensorboard import SummaryWriter


# Define Network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


if __name__ == '__main__':
    do_wandb_logging = True

    writer = SummaryWriter('runs/mnist')

    if do_wandb_logging:
        wandb.init(project='tutorial', entity='friedhelm')

    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # hyper parameters
    input_size = 784 # 28x28
    hidden_size = 500
    num_classes = 10
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001

    if do_wandb_logging:
        wandb.config = {
            'learning_rate': learning_rate,
            'epochs': num_epochs,
            'batch_size': batch_size
        }

    # MNIST
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
        transform=transforms.ToTensor(), download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
        transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
        shuffle=False)

    examples = iter(test_loader)
    samples, labels = examples.next()
    print(samples.shape, labels.shape)

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(samples[i][0], cmap='gray')
    #plt.savefig('results/mnist_example.png')
    img_grid = torchvision.utils.make_grid(samples)
    writer.add_image('mnist_images', img_grid)




    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #writer.add_graph(model, samples.reshape(-1, 28*28))

    # Training loop
    n_total_steps = len(train_loader)
    running_loss = 0.0
    running_correct = 0

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 100, 1, 28, 28
            # 100, 784
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predictions = torch.max(outputs.data, 1)
            running_correct += (predictions == labels).sum().item()

            if (i+1) % 100 == 0:
                print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')
                writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
                writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
                running_loss = 0.0
                running_correct = 0

                if do_wandb_logging:
                    wandb.log({'loss': loss.item()})

    # Test
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)

            # value, index
            _, predictions = torch.max(outputs.data, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'accuarcy = {acc}')
