# Trained for 30 epochs as of 16/10/24
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="Flickr8k/images",
        annotation_file="Flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    load_model = True
    save_model = True
    train_CNN = False

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    step = 0

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        checkpoint = torch.load("my_checkpoint.pth.tar")
        step = load_checkpoint(checkpoint, model, optimizer)
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    model.train()
    for epoch in range(start_epoch, num_epochs):
        if epoch % 2 == 0:
            print(f"Evaluating captions at epoch {epoch}...")
            print_examples(model, device, dataset)

        if epoch % 2 == 0 and epoch != 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch,
            }
            save_checkpoint(checkpoint)
            print(f"Checkpoint saved at epoch {epoch}.")

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()
