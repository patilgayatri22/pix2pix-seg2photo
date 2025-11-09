import argparse, os, random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms

from dataset import SideBySideDataset
from models import UNetGenerator, PatchDiscriminator
from utils import set_seed, denorm, save_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--val_dir", type=str, required=True)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--fm_weight", type=float, default=10.0, help="0 to disable feature matching")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def maybe_compute_fid(gt_dir: Path, pred_dir: Path, device: torch.device) -> Optional[float]:
    try:
        from torch_fidelity import calculate_metrics
        m = calculate_metrics(
            input1=str(gt_dir),
            input2=str(pred_dir),
            cuda=(device.type == "cuda"),
            isc=False, fid=True, kid=False, verbose=False
        )
        return float(m["frechet_inception_distance"])
    except Exception as e:
        print(f"[FID] Skipping (torch-fidelity not available or failed): {e}")
        return None


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir/"checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    sample_dir = outdir/"samples"; sample_dir.mkdir(exist_ok=True)
    pred_dir = outdir/"pred"; pred_dir.mkdir(exist_ok=True)
    gt_dir = outdir/"gt_photos"; gt_dir.mkdir(exist_ok=True)

    # Data
    train_ds = SideBySideDataset(args.train_dir, augment=True,  img_size=256)
    val_ds   = SideBySideDataset(args.val_dir,   augment=False, img_size=256)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Models
    G = UNetGenerator(in_ch=3, out_ch=3).to(device)
    D = PatchDiscriminator(in_ch=6).to(device)

    adv_criterion = nn.BCEWithLogitsLoss()
    l1_criterion  = nn.L1Loss()

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    scheduler_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=100, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=100, gamma=0.5)

    best_fid = float("inf")

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        epG = epD = 0.0

        for cond, target, _ in train_loader:
            cond, target = cond.to(device), target.to(device)

            # --- Train D ---
            opt_D.zero_grad(set_to_none=True)
            real_pair = torch.cat([cond, target], 1)
            pred_real = D(real_pair)

            fake_img = G(cond).detach()
            fake_pair = torch.cat([cond, fake_img], 1)
            pred_fake = D(fake_pair)

            loss_D = 0.5 * (
                adv_criterion(pred_real, torch.ones_like(pred_real)) +
                adv_criterion(pred_fake, torch.zeros_like(pred_fake))
            )
            loss_D.backward()
            opt_D.step()

            # --- Train G ---
            opt_G.zero_grad(set_to_none=True)
            gen_img = G(cond)
            gen_pair = torch.cat([cond, gen_img], 1)
            pred_gen = D(gen_pair)

            loss_G_GAN = adv_criterion(pred_gen, torch.ones_like(pred_gen))
            loss_G_L1  = l1_criterion(gen_img, target) * args.lambda_l1

            loss_FM = 0.0
            if args.fm_weight > 0:
                rf = D.features(real_pair.detach())
                ff = D.features(gen_pair)
                loss_FM = sum(l1_criterion(f, r) for f, r in zip(ff, rf)) * args.fm_weight

            loss_G = loss_G_GAN + loss_G_L1 + loss_FM
            loss_G.backward()
            opt_G.step()

            epG += float(loss_G.item())
            epD += float(loss_D.item())

        scheduler_G.step(); scheduler_D.step()
        avgG = epG / max(len(train_loader), 1)
        avgD = epD / max(len(train_loader), 1)
        print(f"Epoch {epoch}/{args.epochs} | G={avgG:.3f} D={avgD:.3f} | lr={scheduler_G.get_last_lr()[0]:.6f}")

        # Save samples periodically
        if epoch % args.save_every == 0:
            G.eval()
            with torch.no_grad():
                try:
                    cond, target, _ = next(iter(val_loader))
                except StopIteration:
                    cond, target, _ = next(iter(train_loader))
                cond = cond.to(device)
                fake = G(cond).cpu()
                grid = make_grid(
                    torch.cat([denorm(cond.cpu()), denorm(fake), denorm(target)], 0),
                    nrow=cond.size(0)
                )
                img = transforms.ToPILImage()(grid)
                img.save(sample_dir / f"epoch_{epoch:03d}.jpg", quality=95)
            print(f"[Samples] Saved epoch_{epoch:03d}.jpg")

        # Evaluate and track best FID
        if epoch % args.eval_every == 0:
            # clear prev preds
            for p in pred_dir.glob("*.jpg"): p.unlink()
            for p in gt_dir.glob("*.jpg"): p.unlink()

            G.eval()
            with torch.no_grad():
                for cond, target, names in val_loader:
                    cond = cond.to(device)
                    fake = denorm(G(cond)).cpu()
                    # save pred
                    for i, n in enumerate(names):
                        transforms.ToPILImage()(fake[i]).save(pred_dir / n, quality=95)
                    # save GT photo (left half of original image is already target)
                    for i, n in enumerate(names):
                        transforms.ToPILImage()(denorm(target[i])).save(gt_dir / n, quality=95)

            fid = maybe_compute_fid(gt_dir, pred_dir, device)
            if fid is not None:
                print(f"[FID] epoch {epoch}: {fid:.2f}")
                if fid < best_fid:
                    best_fid = fid
                    torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch, "fid": fid},
                               ckpt_dir / "best_model.pt")
                    print(f"[Checkpoint] New best FID {fid:.2f} @ epoch {epoch}")
            else:
                # still save an epoch checkpoint
                torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch},
                           ckpt_dir / f"epoch_{epoch}.pt")

        # Regular checkpoint every save_every
        if epoch % args.save_every == 0:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch},
                       ckpt_dir / f"epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
