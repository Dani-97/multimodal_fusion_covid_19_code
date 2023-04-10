from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as TF

def read_image(image_path, image_width=768, image_height=768):
    dimensions = [image_height, image_width]

    input_image = Image.open(image_path).convert('L')
    input_image_tf = TF.to_tensor(input_image)
    input_image_tf = torchvision.transforms.functional.resize(input_image_tf, dimensions)
    input_image_tf = input_image_tf.unsqueeze_(0)

    return input_image_tf
