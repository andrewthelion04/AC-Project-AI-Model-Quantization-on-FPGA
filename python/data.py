from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def mnist_loaders(data_dir="./data", batch_train=128, batch_test=256):
    tr = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tr)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=tr)
    train_dl = DataLoader(train_ds, batch_size=batch_train, shuffle=True, num_workers=2)
    test_dl  = DataLoader(test_ds, batch_size=batch_test, shuffle=False, num_workers=2)
    return train_ds, test_ds, train_dl, test_dl
