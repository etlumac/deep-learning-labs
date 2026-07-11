import matplotlib.pyplot as plt

def show(imgs, n=16):
    imgs = imgs[:n].reshape(-1,28,28)
    fig, ax = plt.subplots(4,4)
    k = 0
    for i in range(4):
        for j in range(4):
            ax[i,j].imshow(imgs[k], cmap='gray')
            ax[i,j].axis('off')
            k += 1
    plt.show()

def save_imgs(imgs, epoch, folder="results", n=16):
    os.makedirs(folder, exist_ok=True)
    imgs = imgs[:n].reshape(-1,28,28)
    fig, ax = plt.subplots(4,4)
    k = 0
    for i in range(4):
        for j in range(4):
            ax[i,j].imshow(imgs[k], cmap='gray')
            ax[i,j].axis('off')
            k += 1
    plt.savefig(f"{folder}/epoch_{epoch}.png")
    plt.close()
