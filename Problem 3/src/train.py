"""训练循环、学习率衰减与早停。"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, EPOCHS, LR, PATIENCE, LR_PATIENCE, DEVICE, SEED


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, Xtr, ytr, Xva, yva, epochs=EPOCHS, lr=LR,
                patience=PATIENCE, device=DEVICE, verbose=True, seed=SEED):
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=LR_PATIENCE, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(TensorDataset(
        torch.tensor(Xtr), torch.tensor(ytr)), batch_size=BATCH_SIZE, shuffle=True)
    Xva_t = torch.tensor(Xva).to(device)
    yva_t = torch.tensor(yva).to(device)

    history = {"train": [], "val": []}
    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())
        tr_loss = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), yva_t).item()

        history["train"].append(tr_loss)
        history["val"].append(val_loss)
        sched.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, best_state, bad = val_loss, {k: v.cpu().clone()
                                                   for k, v in model.state_dict().items()}, 0
        else:
            bad += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | train {tr_loss:.6f} | val {val_loss:.6f} "
                  f"| lr {sched.get_last_lr()[0]:.1e}")

        if bad >= patience:
            if verbose:
                print(f"  早停于 epoch {epoch}（最佳验证损失 {best_val:.6f}）")
            break

    model.load_state_dict(best_state)
    return model, history
