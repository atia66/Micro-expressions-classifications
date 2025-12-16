import os

from data_processing import create_loader ,split_data
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model import SmallCNN

def train(model, loader, device, epochs, patience):
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    train_loader,val_loader=split_data(loader)
    
    best_val_acc = 0.0
    counter = 0

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total

        # -------- VALIDATION --------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(y).sum().item()
                val_total += y.size(0)

        model.train()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(),"best_model_acc.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early Stopping Triggered — Best Val Accuracy: {best_val_acc:.2f}%")
                break

if __name__ == "__main__":
    time_start = time.time()

    loader = create_loader("./processed_dataset.pt")
    classes=torch.load('./classes.pt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Using device:", device)

    model = SmallCNN(num_classes=len(classes)).to(device)

    train(model, loader, device, epochs=30, patience=3)

    print(f"\nTotal time taken: {time.time() - time_start:.2f} seconds")