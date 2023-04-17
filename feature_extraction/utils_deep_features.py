import copy
import numpy as np
import os
import segmentation_models_pytorch as smp
import timm
import torch
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

    def get_headers(self):
        return self.headers_list

    # This function is used as part of the Adapter Pattern. The same is
    # formulated for radiomics.
    def extract_features(self, input_image_array, input_image_path, \
                                            mask_image_array, mask_image_path, layer='all'):
        return self.extract_deep_features(input_image_array, layer=layer)

# class AlexNet_Deep_Features_Model(Super_Model_Class):
#
#     def __init__(self, **kwargs):
#         self.deep_features_model = torchvision.models.alexnet(pretrained=True)
#         self.device = kwargs['device']
#
#     def extract_deep_features(self, input_data, layer=None):
#         universal_factory = UniversalFactory()
#         kwargs = {}
#         device_chosen = 'Move_To_' + self.device
#         move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)
#
#         input_data = input_data.repeat(1, 3, 1, 1)
#         self.deep_features_model.train(False)
#         self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)
#
#         input_data = move_to_device_module.execute_move_to_device(input_data)
#         # This variable will contain the modified version of the network
#         # architecture that allows to retrieve the features from a certain
#         # layer.
#         modified_deep_features_model = copy.deepcopy(self.deep_features_model)
#         output_features = []
#         output_features.append(modified_deep_features_model(input_data))
#
#         modified_deep_features_model.classifier[5] = Identity()
#         modified_deep_features_model.classifier[6] = Identity()
#         output_features.append(modified_deep_features_model(input_data))
#
#         modified_deep_features_model.classifier[2] = Identity()
#         modified_deep_features_model.classifier[3] = Identity()
#         modified_deep_features_model.classifier[4] = Identity()
#         output_features.append(modified_deep_features_model(input_data))
#
#         it = 0
#         features_merged_together = []
#         for features_set_aux in output_features:
#             features_merged_together+=(features_set_aux.cpu().detach().numpy().tolist()[0])
#             it+=1
#
#         headers_list = []
#         for it in range(0, len(features_merged_together)):
#             headers_list.append('feature_%d'%it)
#         self.headers_list = headers_list
#
#         return features_merged_together
#
#     # The rest of the methods are implemented in the parent class.

class VGG_16_Deep_Features_Model(Super_Model_Class):

    def __init__(self, **kwargs):

        def hook_feat_map(mod, inp, out):
            self.feature_maps.append(out)

        self.deep_features_model = torchvision.models.vgg16(pretrained=True)
        self.deep_features_model.eval()
        self.deep_features_model.classifier[1].register_forward_hook(hook_feat_map)
        self.deep_features_model.classifier[3].register_forward_hook(hook_feat_map)
        self.deep_features_model.classifier[6].register_forward_hook(hook_feat_map)
        self.device = kwargs['device']

    def extract_deep_features(self, input_data, layer='all'):
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

        selected_layer_str = layer.replace('layer_', '')

        if (layer=='all'):
            output_feats_layers = layers_dict['fc6'] + layers_dict['fc7'] + layers_dict['fc8']
        else:
            output_feats_layers = layers_dict[selected_layer_str]

        headers_list = []
        for it in range(0, len(output_feats_layers)):
            headers_list.append('feature_%d'%it)
        self.headers_list = headers_list

        return output_feats_layers

    # The rest of the methods are implemented in the parent class.

class DeiT_Transformer_Model(Super_Model_Class):

    def __init__(self, **kwargs):
        def hook_feat_map(mod, inp, out):
            self.feature_maps.append(out)

        self.deep_features_model = transformers.DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.deep_features_model.pooler.dense.register_forward_hook(hook_feat_map)
        self.deep_features_model.eval()
        self.device = kwargs['device']

    # The parameter 'layer' can be 'layer1', 'layer2', 'layer3'.
    def extract_deep_features(self, input_data, layer='all'):
        self.feature_maps = []

        universal_factory = UniversalFactory()
        kwargs = {}
        device_chosen = 'Move_To_' + self.device
        move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)

        input_data = input_data.repeat(1, 3, 1, 1)
        input_data = move_to_device_module.execute_move_to_device(input_data)

        resize_transf = torchvision.transforms.Resize((224, 224))
        input_data = resize_transf(input_data)

        self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)
        self.deep_features_model(input_data)

        relu_layer = torch.nn.ReLU(inplace=True)

        output_feats_layers = relu_layer(torch.flatten(self.feature_maps[0])).tolist()

        headers_list = []
        for it in range(0, len(output_feats_layers)):
            headers_list.append('feature_%d'%it)
        self.headers_list = headers_list

        return output_feats_layers

class Mixed_Vision_Transformer_Model(Super_Model_Class):

    def __init__(self, **kwargs):
        self.deep_features_model = smp.FPN(encoder_name="mit_b0", encoder_weights="imagenet", in_channels=3, classes=1)
        self.device = kwargs['device']

    # The parameter 'layer' can be 'layer1', 'layer2', 'layer3'.
    def extract_deep_features(self, input_data, layer='all'):
        universal_factory = UniversalFactory()
        kwargs = {}
        device_chosen = 'Move_To_' + self.device
        move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)

        self.deep_features_model.eval()
        self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)

        input_data = input_data.repeat(1, 3, 1, 1)
        input_data = move_to_device_module.execute_move_to_device(input_data)
        output_encoder = self.deep_features_model.encoder(input_data)

        if (layer=='all'):
            output_feats_layers = []

            output_feats_layers.append(np.mean(output_encoder[1].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel())
            output_feats_layers.append(np.mean(output_encoder[2].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel())
            output_feats_layers.append(np.mean(output_encoder[3].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel())
            output_feats_layers.append(np.mean(output_encoder[4].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel())
            output_feats_layers.append(np.mean(output_encoder[5].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel())

            output_feats_layers = np.concatenate(output_feats_layers)
        else:
            layer_int_idx = int(layer.replace('layer', ''))
            output_feats_layers = np.mean(output_encoder[layer_int_idx].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel()

        output_feats_layers = output_feats_layers.tolist()
        headers_list = []
        for it in range(0, len(output_feats_layers)):
            headers_list.append('feature_%d'%it)
        self.headers_list = headers_list

        return output_feats_layers

class DPN_TIMM_Deep_Features_Model(Super_Model_Class):

    def __init__(self, **kwargs):
        self.deep_features_model = timm.create_model('dpn107', pretrained=True)
        self.deep_features_model.eval()
        self.device = kwargs['device']

    # The parameter 'layer' can be 'layer1', 'layer2', 'layer3'.
    def extract_deep_features(self, input_data, layer='all'):
        self.feature_maps = []

        universal_factory = UniversalFactory()
        kwargs = {}
        device_chosen = 'Move_To_' + self.device
        move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)

        input_data = input_data.repeat(1, 3, 1, 1)
        input_data = move_to_device_module.execute_move_to_device(input_data)

        resize_transf = torchvision.transforms.Resize((224, 224))
        input_data = resize_transf(input_data)

        self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)
        model_output = self.deep_features_model(input_data)

        relu_layer = torch.nn.ReLU(inplace=True)

        output_feats_layers = relu_layer(model_output).tolist()[0]

        headers_list = []
        for it in range(0, len(output_feats_layers)):
            headers_list.append('feature_%d'%it)
        self.headers_list = headers_list

        return output_feats_layers

# class DPN_Deep_Features_Model(Super_Model_Class):
#
#     def __init__(self, **kwargs):
#         self.deep_features_model = smp.FPN(encoder_name="dpn68", encoder_weights="imagenet", in_channels=3, classes=1)
#         self.device = kwargs['device']
#
#     def extract_deep_features(self, input_data, layer='all'):
#         universal_factory = UniversalFactory()
#         kwargs = {}
#         device_chosen = 'Move_To_' + self.device
#         move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)
#
#         self.deep_features_model.eval()
#         self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)
#
#         input_data = input_data.repeat(1, 3, 1, 1)
#         input_data = move_to_device_module.execute_move_to_device(input_data)
#         output_encoder = self.deep_features_model.encoder(input_data)
#
#         if (layer=='all'):
#             output_feats_layers = []
#
#             output_feats_layers.append(np.mean(output_encoder[1].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel())
#             output_feats_layers.append(np.mean(output_encoder[2].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel())
#             output_feats_layers.append(np.mean(output_encoder[3].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel())
#             output_feats_layers.append(np.mean(output_encoder[4].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel())
#             output_feats_layers.append(np.mean(output_encoder[5].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel())
#
#             output_feats_layers = np.concatenate(output_feats_layers)
#         else:
#             layer_int_idx = int(layer.replace('layer', ''))
#             output_feats_layers = np.mean(output_encoder[layer_int_idx].cpu().detach().numpy()[0, :, :, :], axis=(1, 2)).ravel()
#
#         output_feats_layers = output_feats_layers.tolist()
#         headers_list = []
#         for it in range(0, len(output_feats_layers)):
#             headers_list.append('feature_%d'%it)
#         self.headers_list = headers_list
#
#         return output_feats_layers

class LaBSE_Deep_Features_Model(Super_Model_Class):

    def __init__(self, **kwargs):
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        self.deep_features_model = transformers.BertModel.from_pretrained("setu4993/LaBSE")
        self.deep_features_model = self.deep_features_model.eval()

        self.device = kwargs['device']

    def extract_deep_features(self, input_data, layer='all'):
        universal_factory = UniversalFactory()
        kwargs = {}
        device_chosen = 'Move_To_' + self.device
        move_to_device_module = universal_factory.create_object(globals(), device_chosen, kwargs)

        self.deep_features_model.eval()
        self.deep_features_model = move_to_device_module.execute_move_to_device(self.deep_features_model)

        output_feats_layers = []
        batches_list = np.array_split(input_data, int(len(input_data)/4))
        for current_batch_aux in batches_list:
            tokenized_sentences = self.tokenizer(current_batch_aux.tolist(), return_tensors="pt", padding=True)
            move_to_device_module.execute_move_to_device(tokenized_sentences)
            with torch.no_grad():
                model_outputs = self.deep_features_model(**tokenized_sentences)

            sentences_embeddings = model_outputs.pooler_output.cpu().detach().numpy()

            if (len(output_feats_layers)==0):
                output_feats_layers = sentences_embeddings
            else:
                output_feats_layers = np.concatenate([output_feats_layers, sentences_embeddings])

        return output_feats_layers

class No_Deep_Features_Model(Super_Model_Class):

    def __init__(self, **kwargs):
        self.headers_list = []

    def extract_deep_features(self, input_data, layer='all'):
        return []
