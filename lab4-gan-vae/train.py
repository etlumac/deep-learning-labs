import numpy as np
from nn.losses import bce, dbce, kl_div
from utils.viz import show, save_imgs
import os

os.makedirs("outputs/reconstructions", exist_ok=True)
os.makedirs("outputs/real", exist_ok=True)

def reparam(mu, logvar):
    eps = np.random.randn(*mu.shape)
    std = np.exp(0.5 * logvar)
    z = mu + std * eps
    return z, eps

def train(encoder, decoder, discriminator, X, epochs=31, lr=5e-3, batch_size=128, adv_weight=0.1):
    N = X.shape[0]

    for e in range(epochs):
        perm = np.random.permutation(N)
        X = X[perm]

        recon_epoch, kl_epoch, adv_epoch, d_epoch = 0, 0, 0, 0

        for i in range(0, N, batch_size):
            x = X[i:i+batch_size]

            mu, logvar = encoder.forward(x)
            z, eps = reparam(mu, logvar)
            x_hat = decoder.forward(z)

            recon = bce(x_hat, x)
            kl = kl_div(mu, logvar)
            beta = min(1.0, e / 10.0)

            #Adversarial losses
            prob_fake = discriminator.forward(x_hat)
            adv_loss_gen = bce(prob_fake, np.ones_like(prob_fake))


            dL_dxhat = dbce(x_hat, x) + adv_weight * dbce(prob_fake, np.ones_like(prob_fake))
            dz = decoder.backward(dL_dxhat)

            std = np.exp(0.5 * logvar)
            dmu = dz + beta * mu
            dlogvar = dz * (0.5 * std * eps) + beta * 0.5 * (np.exp(logvar) - 1)

            encoder.backward(dmu, dlogvar)

            encoder.step(lr)
            decoder.step(lr)

            # Дискриминатор
            prob_real = discriminator.forward(x)
            prob_fake_detach = discriminator.forward(x_hat)
            d_loss = 0.5 * (bce(prob_real, np.ones_like(prob_real)) + bce(prob_fake_detach, np.zeros_like(prob_fake_detach)))

            grad_real = dbce(prob_real, np.ones_like(prob_real))
            grad_fake = dbce(prob_fake_detach, np.zeros_like(prob_fake_detach))
            discriminator.backward(grad_real + grad_fake)
            discriminator.step(lr)

            recon_epoch += recon
            kl_epoch += kl
            adv_epoch += adv_loss_gen
            d_epoch += d_loss


        print(f"Epoch {e}: recon={recon_epoch:.3f} kl={kl_epoch:.3f} adv={adv_epoch:.3f} d_loss={d_epoch:.3f}")

        if e % 5 == 0:
            print("16 реконструированных картинок")
            show(x_hat)
            print("16 реальных картинок")
            show(x[:16])
