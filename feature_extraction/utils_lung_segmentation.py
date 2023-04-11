from collections import OrderedDict
import cv2
import numpy as np
import os, torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, disk, binary_opening, binary_closing

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class LungSegmentationClass():

    def __init__(self):
        pass

    def __convertImage__(self, model_output, border_size = 0):
        # La devolvemos a CPU (en caso de que no estuviera) y la reescalamos al tamaño original con
        # interpolación bicúbica para evitar pixelado
        PIL_IMAGE = transforms.ToPILImage()(model_output.squeeze(0).to('cpu'))
        current_image_size = model_output.squeeze(0).shape[1:]

        # Apertura antes de reescalado para optimizar
        NUMPY_IMAGE = binary_opening(np.array(PIL_IMAGE),disk(3))
        NUMPY_IMAGE = binary_closing(NUMPY_IMAGE,disk(3)).astype('uint8')

        # Rellenamos huecos dentro de pulmones
        # Rellenaremos el background y añadiremos el inverso para que queden rellenos los huecos.
        h, w = NUMPY_IMAGE.shape[:2]

        # La primera y última filas se dejan vacías para los pulmones que abarquen toda la imagen. De esta manera, no dejarán hueco entre ellos
        # que estropee el floodFill
        NUMPY_IMAGE[0,:] = 0
        NUMPY_IMAGE[h-1,:] = 0

        mask = np.zeros((h+2, w+2), np.uint8)
        ONLY_HOLES = NUMPY_IMAGE.copy().astype('uint8')
        cv2.floodFill(ONLY_HOLES, mask, (0,0), 255)

        ONLY_HOLES = cv2.bitwise_not(ONLY_HOLES)
        NUMPY_IMAGE = NUMPY_IMAGE | ONLY_HOLES

        PIL_IMAGE = Image.fromarray(NUMPY_IMAGE).resize(current_image_size, Image.BICUBIC)

        # Binarizamos image para eliminar el difuminado de la interpolación
        NUMPY_IMAGE = np.array(PIL_IMAGE)
        NUMPY_IMAGE = NUMPY_IMAGE > threshold_otsu(NUMPY_IMAGE)

        # Si nos pide bordes, erosionamos y restamos a la imagen original
        if border_size > 0:
            ERODED_IMAGE = erosion(NUMPY_IMAGE,disk(border_size))
            NUMPY_IMAGE = NUMPY_IMAGE & (np.invert(ERODED_IMAGE))


        # Quedarnos con las dos componentes más grandes si es posible.
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(NUMPY_IMAGE.astype('uint8'), connectivity=4)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        NUMPY_IMAGE = np.zeros((output.shape))

        if len(sizes) <= 2:
            for i in range(0, nb_components):
                NUMPY_IMAGE[output == i + 1] = 255
        else:
            reordered_list = sorted(sizes,reverse=True)

            i = list(sizes).index(reordered_list[0])
            NUMPY_IMAGE[output == i + 1] = 255
            i = list(sizes).index(reordered_list[1])
            NUMPY_IMAGE[output == i + 1] = 255

        return Image.fromarray(NUMPY_IMAGE).convert('RGB')

    def obtain_lung_segmentation(self, images_root_dir, results_dir_root, \
            trained_model_path = './imaging_features/trained_segmentation_model.pt', device='cuda:0'):
        print('Obtaining lung segmentation with model stored at %s..'%trained_model_path)

        current_network = UNet()
        current_network.load_state_dict(torch.load(trained_model_path))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        current_network = current_network.to(device)
        current_network.train()

        image_names_list = os.listdir(images_root_dir)

        for current_image_name_aux in image_names_list:
            print('++++ Processing image %s...'%current_image_name_aux)
            current_image = Image.open(os.path.join(images_root_dir, current_image_name_aux)).resize((256, 256)).convert('RGB', colors=256)
            segmentation_mask_image_name = current_image_name_aux.replace('.png', '_segmentation.png')
            current_mask_image_path = os.path.join(results_dir_root, segmentation_mask_image_name)

            tensor_image = (transforms.ToTensor()(current_image)*255).type(torch.uint8)
            tensor_image = (transforms.functional.equalize(tensor_image).type(torch.float)/255)
            tensor_image = tensor_image.unsqueeze(0)

            image = tensor_image.to(device)

            output = current_network(image).data
            output_postprocessed = self.__convertImage__(output)
            output_postprocessed.save('%s/%s'%(results_dir_root, current_image_name_aux))
