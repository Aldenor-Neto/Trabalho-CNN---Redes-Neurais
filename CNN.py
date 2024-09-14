import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Definição da CNN
class Net(nn.Module):
    def __init__(self, in_channels, num_conv_layers, num_filters, image_size):
        super(Net, self).__init__()
        layers = []
        channels = in_channels
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(channels, num_filters, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            channels = num_filters
        self.conv = nn.Sequential(*layers)
        self.fc_input_size = num_filters * (image_size // 2**num_conv_layers)**2
        self.fc = nn.Linear(self.fc_input_size, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc_input_size)
        x = self.fc(x)
        return x

# Função para treinar o modelo
def train_model(net, trainloader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_per_epoch = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(trainloader)
        loss_per_epoch.append(epoch_loss)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
    return loss_per_epoch

# Função para testar o modelo
def test_model(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}')
    return accuracy

# Função para rodar os experimentos
def run_experiments(configs, epochs=10):
    transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist, batch_size=64, shuffle=True)

    testset_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    testloader_mnist = torch.utils.data.DataLoader(testset_mnist, batch_size=64, shuffle=False)

    trainset_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    trainloader_cifar = torch.utils.data.DataLoader(trainset_cifar, batch_size=64, shuffle=True)

    testset_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
    testloader_cifar = torch.utils.data.DataLoader(testset_cifar, batch_size=64, shuffle=False)

    loss_curves = {}
    for config in configs:
        print(f"\nTreinando para configuração: {config}")
        if config['dataset'] == 'MNIST':
            trainloader, testloader = trainloader_mnist, testloader_mnist
        elif config['dataset'] == 'CIFAR10':
            trainloader, testloader = trainloader_cifar, testloader_cifar

        net = Net(config['in_channels'], config['num_conv_layers'], config['num_filters'], config['image_size'])
        loss_per_epoch = train_model(net, trainloader, epochs)
        accuracy = test_model(net, testloader)

        # Salva as curvas de perda para cada configuração
        loss_curves[config['dataset'] + f"_{config['num_filters']}"] = loss_per_epoch

        # Plot gráfico individual para cada treino
        plt.figure()
        plt.plot(range(1, epochs + 1), loss_per_epoch)
        plt.title(f"Loss Curve - {config['dataset']} - Filters: {config['num_filters']}")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    # Plot gráfico geral com todas as curvas
    plt.figure()
    for label, losses in loss_curves.items():
        plt.plot(range(1, epochs + 1), losses, label=label)
    plt.title('Combined Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Configurações para MNIST e CIFAR-10
configs = [
    {'dataset': 'MNIST', 'in_channels': 1, 'image_size': 28, 'num_conv_layers': 2, 'num_filters': 32},
    {'dataset': 'MNIST', 'in_channels': 1, 'image_size': 28, 'num_conv_layers': 3, 'num_filters': 64},
    {'dataset': 'CIFAR10', 'in_channels': 3, 'image_size': 32, 'num_conv_layers': 2, 'num_filters': 32},  # Adicionada configuração faltante
    {'dataset': 'CIFAR10', 'in_channels': 3, 'image_size': 32, 'num_conv_layers': 3, 'num_filters': 64}
]

# Executa os experimentos
run_experiments(configs, epochs=10)
