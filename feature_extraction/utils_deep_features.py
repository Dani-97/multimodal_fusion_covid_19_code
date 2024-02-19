import copy
import numpy as np
import os
from PIL import Image
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import transformers

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

        output_data = input_data.to('cuda:0')

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

    # This function is used as part of the Adapter Pattern. The same is
    # formulated for radiomics.
    def extract_features(self, input_image_array):
        return self.extract_deep_features(input_image_array)

class Super_ProgPred_Class():

    def __init__(self):
        pass

    def apply_progression_model(self, input_data):
        pass


class No_Deep_Features_Model(Super_Model_Class):

    def __init__(self, **kwargs):
        self.headers_list = []

    def extract_deep_features(self, input_data):
        return [], []

class VGG_16_Deep_Features_Model(Super_Model_Class):

    def __init__(self, **kwargs):

        def hook_feat_map(mod, inp, out):
            self.feature_maps.append(out)

        self.deep_features_model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
        self.deep_features_model.eval()
        self.deep_features_model.classifier[1].register_forward_hook(hook_feat_map)
        self.deep_features_model.classifier[3].register_forward_hook(hook_feat_map)
        self.deep_features_model.classifier[6].register_forward_hook(hook_feat_map)
        self.device = kwargs['device']
        self.layer = kwargs['layer']

    def extract_deep_features(self, input_data):
        self.feature_maps = []

        universal_factory = UniversalFactory()
        kwargs = {}
        device_chosen = 'Move_To_' + self.device
        move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)

        self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)

        input_data = input_data.repeat(1, 3, 1, 1)
        input_data = move_to_device_module.execute_move_to_device(input_data)

        # The output is not necessary. The inference call is only needed to
        # retrieve the feature maps.
        self.deep_features_model(input_data)

        relu_layer = torch.nn.ReLU(inplace=True)

        layers_dict = {}
        layers_dict['fc6'] = torch.flatten(self.feature_maps[0]).cpu().tolist()
        layers_dict['fc7'] = torch.flatten(self.feature_maps[1]).cpu().tolist()
        layers_dict['fc8'] = torch.flatten(relu_layer(self.feature_maps[2])).cpu().tolist()

        selected_layer_str = self.layer.replace('layer_', '')

        if (self.layer=='all'):
            output_feats_layers = layers_dict['fc6'] + layers_dict['fc7'] + layers_dict['fc8']
        else:
            output_feats_layers = layers_dict[selected_layer_str]

        headers_list = []
        for it in range(0, len(output_feats_layers)):
            headers_list.append('feature_%d'%it)

        return headers_list, output_feats_layers

    # The rest of the methods are implemented in the parent class.

class LSTM_ProgPred_Model(Super_ProgPred_Class):

    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs['device']
        self.nof_imaging_features = kwargs['nof_imaging_features']
        self.lstm_model = nn.LSTM(self.nof_imaging_features, self.nof_imaging_features)

        universal_factory = UniversalFactory()
        kwargs = {}
        device_chosen = 'Move_To_' + self.device
        self.move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)

        self.lstm_model = self.move_to_device_module.execute_move_to_device(self.lstm_model)

    def apply_progression_model(self, input_df):
        imaging_features_headers = list(filter(lambda input_value: input_value.find('feature_')!=-1, input_df.columns))
        nof_imaging_features = len(imaging_features_headers)

        deep_features = input_df[imaging_features_headers].values

        for current_image_features_aux in deep_features:
            deep_features_tensor = torch.FloatTensor(current_image_features_aux).unsqueeze(0).unsqueeze(0)
            deep_features_tensor = self.move_to_device_module.execute_move_to_device(deep_features_tensor)

            rand_hidden_state = self.move_to_device_module.execute_move_to_device(torch.randn(1, 1, nof_imaging_features))
            hidden = (rand_hidden_state, rand_hidden_state)

            lstm_output, hidden = self.lstm_model(deep_features_tensor, hidden)

            final_output = lstm_output.tolist()[0][0]

        return final_output