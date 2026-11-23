import tqdm
from collections import defaultdict
import torch
import pathlib
import os 
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

def train(model, epochs, optimizer, criterion, train_loader, val_loader, device = None):
    history = defaultdict(list)

    for epoch in range(epochs):
        with tqdm.tqdm(enumerate(train_loader), total=len(train_loader)) as bar:
            model.train()
            bar.set_description(f"Epoch {epoch:02d}")

            loss_tmp = 0
            accuracy_tmp = 0
            for i, data in bar:
                # training
                inputs, labels = data

                if device is not None:
                    inputs, labels = inputs.to(device=device).float(), labels.to(device=device).long()

                # Forward to get output
                outputs = model(inputs)
                
                # Calculate Loss
                loss = criterion(outputs, labels)
                accuracy = (outputs.argmax(dim=1) == labels).float().mean()

                loss_tmp += float(loss.detach())
                accuracy_tmp += float(accuracy)
                
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad() 

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                if i < len(train_loader) - 1:
                    bar.set_postfix({"loss": loss_tmp / (i + 1), "accuracy": accuracy_tmp / (i + 1)})
                else:
                    # validation after last batch

                    history["loss"].append(loss_tmp / len(train_loader))
                    history["accuracy"].append(accuracy_tmp / len(train_loader))
                
                    model.eval()
                    with torch.no_grad():
                        loss_tmp = 0
                        accuracy_tmp = 0
                        for data in val_loader:
                            inputs, labels = data

                            if device is not None:
                                inputs, labels = inputs.to(device=device).float(), labels.to(device=device).long()

                            outputs = model(inputs)
                            
                            loss = criterion(outputs, labels)   
                            accuracy = (outputs.argmax(dim=1) == labels).float().mean()

                            loss_tmp += float(loss)
                            accuracy_tmp += float(accuracy)
                        
                        history["loss_val"].append(loss_tmp / len(val_loader))
                        history["accuracy_val"].append(accuracy_tmp / len(val_loader))
                        
                    bar.set_postfix({"loss": history["loss"][-1], "accuracy": history["accuracy"][-1], 
                                    "loss_val": history["loss_val"][-1], "accuracy_val": history["accuracy_val"][-1]})
    return history

def train2(model: torch.nn.Module, epochs: int, optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.modules.loss._Loss, train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader, device: torch.cuda.device = None, 
          scheduler: torch.optim.lr_scheduler.LRScheduler = None, clip_grads: bool = False,
          save_best_path: str = None):
    
    history = defaultdict(list)
    best_accuracy = 0.0

    # create save path
    if save_best_path is not None:
        pathlib.Path(os.path.dirname(save_best_path)).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # training       
        model.train()        
        loss_tmp = 0
        accuracy_tmp = 0

        with tqdm.tqdm(enumerate(train_loader), total=len(train_loader)) as bar:
            bar.set_description(f"Epoch {epoch+1:02d} / {epochs:02d}")
            for i, (inputs, labels) in bar:
                if device is not None:
                    # inputs float, labels long (CrossEntropy expects LongTensor of shape (B,))
                    inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
                    labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
                else:
                    inputs = inputs.float()
                    labels = labels.long()
                
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad() 

                if inputs.isnan().any():
                    print("INPUT CONTAINS NAN. SKIPPING BATCH")
                    continue

                # Forward to get output
                outputs = model(inputs)
                
                # Calculate Loss
                loss = criterion(outputs, labels)
                # accuracy: compare predicted class (argmax logits) with integer labels
                batch_acc = (outputs.argmax(dim=1) == labels).float().mean()

                # Getting gradients w.r.t. parameters
                loss.backward()

                # clip grads
                if clip_grads:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                # Updating parameters
                optimizer.step()

                loss_tmp += float(loss.detach())
                accuracy_tmp += float(batch_acc.detach())

                bar.set_postfix({"loss": loss_tmp / (i + 1), "accuracy": accuracy_tmp / (i + 1)})
        
        # add to history
        history["loss"].append(loss_tmp / len(train_loader))
        history["accuracy"].append(accuracy_tmp / len(train_loader))

        # evaluate model
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            with tqdm.tqdm(enumerate(val_loader), total=len(val_loader)) as vbar:
                vbar.set_description("  - eval")
                for j, (inputs, labels) in vbar:
                    if device is not None:
                        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
                        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
                    else:
                        inputs = inputs.float()
                        labels = labels.long()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    batch_acc = (outputs.argmax(dim=1) == labels).float().mean()

                    val_loss += float(loss.detach())
                    val_acc += float(batch_acc.detach())

                    vbar.set_postfix({"loss": history["loss"][-1], "accuracy": history["accuracy"][-1],
                                    "val_loss": val_loss / (j + 1), "val_accuracy": val_acc / (j + 1)})

        val_loss /= len(val_loader)
        val_acc  /= len(val_loader)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
            
        # update best accuracy
        if save_best_path is not None and val_acc > best_accuracy:
            best_accuracy = val_acc
            # save state_dict
            torch.save(model.state_dict(), save_best_path)
            print(f"Saved best model with val_acc={best_accuracy:.4f} to {save_best_path}")

        # update lr scheduler
        if scheduler is not None:
            # wenn ReduceLROnPlateau: übergebe val_loss
            try:
                scheduler.step(val_loss)
            except Exception:
                scheduler.step()
        
        print()
    return history

def test(path, transform, device, image_model, CLASSES):
    # Load and display image
    img_bgr = cv2.imread(path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Convert image to tensor
    img_tensor = transform(img).unsqueeze(0).to(device=device, dtype=torch.float32)

    # Take prediction
    image_model.eval()
    with torch.no_grad():
        logits = image_model(img_tensor)
        probs = F.softmax(logits, dim=1)

    # Evaluate Score
    probs = probs.squeeze(0)  # shape (C,)
    pred_idx = int(probs.argmax().item())
    confidence = float(probs.max().item()) * 100.0

    print(f"This image is a {CLASSES[pred_idx]} with a {confidence:.2f}% confidence.")
    return pred_idx, confidence