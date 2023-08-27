from src.dataset import dataset_wrapper
from torch.utils.data import DataLoader
dataset=dataset_wrapper('cifar10', 32)

dataloader=DataLoader(dataset, batch_size=32, shuffle=True)
dl=iter(dataloader)
print(next(dl)[0].shape)