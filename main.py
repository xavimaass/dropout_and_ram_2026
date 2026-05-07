import numpy as np

from config import DropoutResNetConfig
from src.resnet.dropout_model import DropoutResNet


def main():
    """Train a dropout ResNet on a simple random regression task."""

    cfg = DropoutResNetConfig(D=16, M=8, L=4, q=0.8, alpha=1.0, seed=42)

    model = DropoutResNet(
        D=cfg.D,
        M=cfg.M,
        L=cfg.L,
        q=cfg.q,
        alpha=cfg.alpha,
        seed=cfg.seed,
    )

    rng = np.random.default_rng(0)
    N = 32          # batch size
    tau = 1e-3      # learning rate
    n_steps = 200   # training steps

    # Random regression target: map D-dim input to D-dim output
    X = rng.standard_normal((cfg.D, N))
    Y = rng.standard_normal((cfg.D, N))

    print(f"Training dropout ResNet  D={cfg.D}  M={cfg.M}  L={cfg.L}  q={cfg.q}")
    for step in range(n_steps):
        loss = model.step(X, Y, tau=tau)
        if (step + 1) % 50 == 0:
            print(f"  step {step + 1:4d}  loss = {loss:.6f}")

    print("Done.")


if __name__ == "__main__":
    main()
