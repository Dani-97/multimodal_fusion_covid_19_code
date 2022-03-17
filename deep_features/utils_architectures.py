import copy
import numpy as np
import torch
from torchvision import models

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

    def execute_move_to_device(self, input_data):
        print('++++ A variable is being kept in CPU')

        return input_data

# This class is used to keep a PyTorch tensor in CPU.
class Move_To_CUDA(Move_To_CPU):

    def __init__(self):
        pass

    def execute_move_to_device(self, input_data):
        print('++++ A variable is being transferred to a CUDA device')
        output_data = input_data.cuda()

        return output_data

class AlexNet_Deep_Features_Model():

    def __init__(self, **kwargs):
        self.deep_features_model = models.alexnet()
        self.device = kwargs['device']

    def extract_deep_features(self, input_data):
        universal_factory = UniversalFactory()
        kwargs = {}
        device_chosen = 'Move_To_' + self.device
        move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)

        self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)
        # This variable will contain the features of the target layers.
        deep_features_list = []
        for batch_features, _ in input_data:
            batch_features = move_to_device_module.execute_move_to_device(batch_features)
            # This variable will contain the modified version of the network
            # architecture that allows to retrieve the features from a certain
            # layer.
            modified_deep_features_model = copy.deepcopy(self.deep_features_model)
            outputs_fc_1000 = modified_deep_features_model(batch_features)

            modified_deep_features_model.classifier[5] = Identity()
            modified_deep_features_model.classifier[6] = Identity()
            outputs_fc_2_4096 = modified_deep_features_model(batch_features)

            modified_deep_features_model.classifier[2] = Identity()
            modified_deep_features_model.classifier[3] = Identity()
            modified_deep_features_model.classifier[4] = Identity()
            outputs_fc_1_4096 = modified_deep_features_model(batch_features)

            print('FC 1000 -> ', np.shape(outputs_fc_1000))
            print('FC1 4096 -> ', np.shape(outputs_fc_1_4096))
            print('FC2 4096 -> ', np.shape(outputs_fc_2_4096))

class VGG_16_Deep_Features_Model():

    def __init__(self, **kwargs):
        self.deep_features_model = models.vgg16(pretrained=True)
        self.device = kwargs['device']

    def extract_deep_features(self, input_data):
        universal_factory = UniversalFactory()
        kwargs = {}
        device_chosen = 'Move_To_' + self.device
        move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)

        self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)
        # This variable will contain the features of the target layers.
        deep_features_list = []
        for batch_features, _ in input_data:
            batch_features = move_to_device_module.execute_move_to_device(batch_features)
            # This variable will contain the modified version of the network
            # architecture that allows to retrieve the features from a certain
            # layer.
            modified_deep_features_model = copy.deepcopy(self.deep_features_model)
            outputs_fc_1000 = modified_deep_features_model(batch_features)

            modified_deep_features_model.classifier[4] = Identity()
            modified_deep_features_model.classifier[5] = Identity()
            modified_deep_features_model.classifier[6] = Identity()
            outputs_fc_2_4096 = modified_deep_features_model(batch_features)

            modified_deep_features_model.classifier[1] = Identity()
            modified_deep_features_model.classifier[2] = Identity()
            modified_deep_features_model.classifier[3] = Identity()
            outputs_fc_1_4096 = modified_deep_features_model(batch_features)

            print('FC 1000 -> ', np.shape(outputs_fc_1000))
            print('FC1 4096 -> ', np.shape(outputs_fc_1_4096))
            print('FC2 4096 -> ', np.shape(outputs_fc_2_4096))
