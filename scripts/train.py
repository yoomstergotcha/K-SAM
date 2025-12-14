import torch
from tqdm import tqdm

from utils.losses import (
    l1_loss, d_hinge_loss, g_hinge_loss,
    identity_weight, cosine_dist_per_sample, age_to_bin
)


# =========================
# Stage 2: Reconstruction
# =========================
def train_stage2(
    G, train_loader, val_loader,
    opt_G, device,
    epochs=10
):
    for ep in range(1, epochs+1):
        G.train()
        train_loss = 0.0

        for x_src, age_src, _, _, _ in train_loader:
            x_src = x_src.to(device)
            age_src = age_src.to(device)

            x_hat, _ = G(x_src, age_src, age_src)
            loss = l1_loss(x_hat, x_src)

            opt_G.zero_grad()
            loss.backward()
            opt_G.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- validation ----
        G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_src, age_src, _, _, _ in val_loader:
                x_src = x_src.to(device)
                age_src = age_src.to(device)
                x_hat, _ = G(x_src, age_src, age_src)
                val_loss += l1_loss(x_hat, x_src).item()

        val_loss /= len(val_loader)

        print(
            f"[Stage2] ep {ep}/{epochs} | "
            f"train L1={train_loss:.4f} | val L1={val_loss:.4f}"
        )


# =========================
# Stage 3: GAN + Geometry
# =========================
def train_stage3(
    G, D,
    ssrnet, id_net, lpips_fn,
    age_landmark_prototypes,
    train_loader,
    opt_G, opt_D,
    device,
    epochs=5,
    W_GAN=1.0,
    W_AGE=0.5,
    W_ID=0.2,
    W_LM_AGE=80.0,
    W_LM_ID=2.0,
    W_PERC=0.02
):
    ssrnet.eval()
    id_net.eval()

    for ep in range(1, epochs+1):
        sum_age = 0.0

        for x_src, age_src, age_tgt, _, lm_src in tqdm(train_loader):
            x_src = x_src.to(device)
            age_src = age_src.to(device)
            age_tgt = age_tgt.to(device)
            lm_src = lm_src.to(device)

            # ---- D update ----
            with torch.no_grad():
                x_fake, _ = G(x_src, age_src, age_tgt)

            loss_D = d_hinge_loss(D(x_src), D(x_fake))
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---- G update ----
            x_fake, lm_hat = G(x_src, age_src, age_tgt)

            loss_gan = g_hinge_loss(D(x_fake))

            age_pred = ssrnet(x_fake)
            loss_age = torch.abs(age_pred - age_tgt.squeeze(1)).mean()

            loss_perc = lpips_fn(x_fake, x_src).mean()

            with torch.no_grad():
                id_src = id_net(x_src)
            id_hat = id_net(x_fake)

            age_gap = (age_tgt - age_src).abs().view(-1)
            w = identity_weight(age_gap).to(device)

            loss_id = (w * cosine_dist_per_sample(id_hat, id_src)).mean()
            loss_lm_id = (w.view(-1,1,1) * (lm_hat - lm_src).abs()).mean()

            age_bin = age_to_bin(age_tgt)
            proto = torch.stack(
                [age_landmark_prototypes[int(b)] for b in age_bin.tolist()]
            ).to(device)

            loss_lm_age = torch.abs(lm_hat - proto).mean()

            loss_G = (
                W_GAN * loss_gan +
                W_AGE * loss_age +
                W_PERC * loss_perc +
                W_ID * loss_id +
                W_LM_ID * loss_lm_id +
                W_LM_AGE * loss_lm_age
            )

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            sum_age += loss_age.item()

        print(
            f"[Stage3] ep {ep}/{epochs} | "
            f"AgeL1={sum_age/len(train_loader):.3f}"
        )
