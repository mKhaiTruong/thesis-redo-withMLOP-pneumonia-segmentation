import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from training.utils.metrics.iou import compute_iou

def train_one_epoch(model, loader, criterion, optimizer, scheduler, epoch, device):
    model.train()
    total_loss = 0.0
    scaler = GradScaler(device, enabled=True)
    
    progBar = tqdm(loader, desc=f'Epoch {epoch} [TRAIN]', leave=False)

    for input in progBar:
        images, masks = input['image'], input['mask']
        images, masks = images.to(device), masks.to(device)
        
        with autocast(device_type=device.type):
            loss = criterion(model(images), masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        total_loss += loss.item()
        progBar.set_postfix(loss=loss.item()) 
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, epoch, device):
    model.eval()
    total_loss, iou = 0.0, []
    
    progBar = tqdm(loader, desc=f'Epoch {epoch} [VALID]', leave=False)
    
    for input in progBar:
        images, masks = input['image'], input['mask']
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, masks).item()
        iou.append(compute_iou(outputs, masks))
        
        progBar.set_postfix(iou=iou[-1])
            
    return total_loss / len(loader), sum(iou) / len(iou)


# Sanity check
def train_one_epoch_sanity_check(model, loader, criterion, optimizer, scheduler, epoch, device):
    model.train()
    batch = next(iter(loader))
    scaler = GradScaler(device, enabled=True)
    
    images = batch['image'].to(device)
    masks = batch['mask'].to(device)
    
    print(images.shape)
    print(masks.shape)
    print(masks.unique())
    
    preds = torch.sigmoid(model(images))
    print(preds.shape)
    print(preds.min().item(), preds.max().item())
    print((preds > 0.5).float().sum().item())
    print(masks.sum().item())

    for i in range(20):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            loss = criterion(model(images), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        preds = torch.sigmoid(model(images))
        iou = compute_iou(preds, masks)
        print(f"Step {i} | Loss {loss.item():.4f} | IOU {iou:.4f}")