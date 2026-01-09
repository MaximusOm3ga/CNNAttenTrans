import torch
import torch.nn as nn
import torch.optim as optim
from Transformer.proposed.modelCNN.model import AutoencoderTokenExtractor
from Transformer.proposed.modelCNN.data_load import train_loader, val_loader

LR = 1e-3
EPOCHS = 600
PATCH_SIZE = 4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoencoderTokenExtractor().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.SmoothL1Loss()

best_val_loss = float("inf")
epochs = EPOCHS
patch_size = PATCH_SIZE

for epoch in range(epochs):
    model.train()
    train_loss_sum = 0.0
    train_elems = 0

    for (x,) in train_loader:
        x = x.to(device)
        B, T, D = x.shape

        if T < patch_size:
            continue

        num_patches = T // patch_size
        T_eff = num_patches * patch_size
        x_eff = x[:, :T_eff, :]

        patches = [x_eff[:, i * patch_size : (i + 1) * patch_size, :] for i in range(num_patches)]
        if not patches:
            continue
        patches_target = torch.stack(patches, dim=1)

        reconstructed, _ = model(x_eff)
        if reconstructed.shape != patches_target.shape:
            raise RuntimeError(
                f"Shape mismatch: reconstructed {reconstructed.shape} vs target {patches_target.shape}. "
                f"Check encoder/decoder patch_size and config.cnn.patch_size."
            )

        loss = criterion(reconstructed, patches_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        elems = reconstructed.numel()
        train_loss_sum += loss.item() * elems
        train_elems += elems

    train_loss = train_loss_sum / max(train_elems, 1)
    model.eval()
    val_loss_sum = 0.0
    val_elems = 0

    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(device)
            B, T, D = x.shape

            if T < patch_size:
                continue
            num_patches = T // patch_size
            T_eff = num_patches * patch_size
            x_eff = x[:, :T_eff, :]
            patches = [x_eff[:, i * patch_size : (i + 1) * patch_size, :] for i in range(num_patches)]
            if not patches:
                continue
            patches_target = torch.stack(patches, dim=1)
            reconstructed, _ = model(x_eff)
            if reconstructed.shape != patches_target.shape:
                raise RuntimeError(
                    f"Shape mismatch (val): reconstructed {reconstructed.shape} vs target {patches_target.shape}. "
                    f"Check encoder/decoder patch_size and config.cnn.patch_size."
                )
            loss = criterion(reconstructed, patches_target)
            elems = reconstructed.numel()
            val_loss_sum += loss.item() * elems
            val_elems += elems

    val_loss = val_loss_sum / max(val_elems, 1)
    print(f"Epoch {epoch}: Train {train_loss:.6f} | Val {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.encoder.state_dict(), "trained_encoder_new.pth")
        print("Saved best encoder")

    scheduler.step()

print("Training complete")
