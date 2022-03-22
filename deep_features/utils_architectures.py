import copy
import numpy as np
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as TF

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

# Auxiliar class for the deep features extraction from the models.
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# This class is used to move a PyTorch tensor to GPU.
class Move_To_CPU():

    def __init__(self):
        pass

    def execute_move_to_device(self, input_data, logging=False):
        if (logging):
            print('++++ A variable is being kept in CPU')

        return input_data

# This class is used to keep a PyTorch tensor in CPU.
class Move_To_CUDA(Move_To_CPU):

    def __init__(self):
        pass

    def execute_move_to_device(self, input_data, logging=False):
        if (logging):
            print('++++ A variable is being transferred to a CUDA device')

        output_data = input_data.cuda()

        return output_data

class Super_Model_Class():

    def __init__(self):
        pass

    def read_image(self, image_path, image_width=768, image_height=768):
        dimensions = [image_height, image_width]

        input_image = Image.open(image_path)
        input_image_tf = TF.to_tensor(input_image)
        input_image_tf = torchvision.transforms.functional.resize(input_image_tf, dimensions)
        input_image_tf = input_image_tf.unsqueeze_(0)

        return input_image_tf

class AlexNet_Deep_Features_Model(Super_Model_Class):

    def __init__(self, **kwargs):
        self.deep_features_model = torchvision.models.alexnet()
        self.device = kwargs['device']

    def extract_deep_features(self, input_data):
        universal_factory = UniversalFactory()
        kwargs = {}
        device_chosen = 'Move_To_' + self.device
        move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)

        self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)

        input_data = move_to_device_module.execute_move_to_device(input_data)
        # This variable will contain the modified version of the network
        # architecture that allows to retrieve the features from a certain
        # layer.
        modified_deep_features_model = copy.deepcopy(self.deep_features_model)
        output_features = []
        output_features.append(modified_deep_features_model(input_data))

        modified_deep_features_model.classifier[5] = Identity()
        modified_deep_features_model.classifier[6] = Identity()
        output_features.append(modified_deep_features_model(input_data))

        modified_deep_features_model.classifier[2] = Identity()
        modified_deep_features_model.classifier[3] = Identity()
        modified_deep_features_model.classifier[4] = Identity()
        output_features.append(modified_deep_features_model(input_data))

        it = 0
        features_merged_together = []
        for features_set_aux in output_features:
            features_merged_together+=(features_set_aux.cpu().detach().numpy().tolist()[0])
            it+=1

        return features_merged_together

    # The rest of the methods are implemented in the parent class.

class VGG_16_Deep_Features_Model(Super_Model_Class):

    def __init__(self, **kwargs):
        self.deep_features_model = torchvision.models.vgg16(pretrained=True)
        self.device = kwargs['device']

    def extract_deep_features(self, input_data):
        universal_factory = UniversalFactory()
        kwargs = {}
        device_chosen = 'Move_To_' + self.device
        move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)

        self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)

        input_data = move_to_device_module.execute_move_to_device(input_data)
        # This variable will contain the modified version of the network
        # architecture that allows to retrieve the features from a certain
        # layer.
        modified_deep_features_model = copy.deepcopy(self.deep_features_model)
        output_features = []
        output_features.append(modified_deep_features_model(input_data))

        modified_deep_features_model.classifier[4] = Identity()
        modified_deep_features_model.classifier[5] = Identity()
        modified_deep_features_model.classifier[6] = Identity()
        output_features.append(modified_deep_features_model(input_data))

        modified_deep_features_model.classifier[1] = Identity()
        modified_deep_features_model.classifier[2] = Identity()
        modified_deep_features_model.classifier[3] = Identity()
        output_features.append(modified_deep_features_model(input_data))

        it = 0
        features_merged_together = []
        for features_set_aux in output_features:
            features_merged_together+=(features_set_aux.cpu().detach().numpy().tolist()[0])
            it+=1

        return features_merged_together

        # The rest of the methods are implemented in the parent class.
