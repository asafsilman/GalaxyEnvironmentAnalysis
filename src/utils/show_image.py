from PIL import Image
from matplotlib import pyplot as plt

def show_image(data_set):
    img, label = next(iter(data_set.take(1)))
    
    img = img.numpy()[0].reshape((50,50)) * 255
    lbl = label.numpy[0]
    image =  Image.fromarray(img).convert("L")

    plt.title(f"Class = {lbl}")
    plt.imshow(image)
