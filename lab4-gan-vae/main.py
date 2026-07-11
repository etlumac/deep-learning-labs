from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator
from utils.data import load_mnist
from train import train

X = load_mnist()

enc = Encoder()
dec = Decoder()
disc = Discriminator()

train(enc, dec, disc, X)
