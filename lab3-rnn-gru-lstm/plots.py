from PIL import Image
import os

def show_saved():
    for name in ['results/train_loss.png', 'results/pred_vs_true.png']:
        if os.path.exists(name):
            try:
                img = Image.open(name)
                img.show()
            except Exception as e:
                print('Cannot open', name, '->', e)
        else:
            print('Not found:', name)

if __name__ == '__main__':
    show_saved()
