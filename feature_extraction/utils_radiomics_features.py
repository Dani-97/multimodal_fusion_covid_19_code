import cv2
import numpy as np
import os
from radiomics import featureextractor
import SimpleITK as sitk
import six

class Radiomics_Features_Super_Class():

    def __init__(self, **kwargs):
        self.headers_list = []

    def get_headers(self):
        return self.headers_list

    def obtain_features(self, input_image_path, mask_image_path):
        raise NotImplementedError('The method obtain_features must be implemented by the child class!')

    def extract_features(self, input_image_array, input_image_path, \
                                                mask_image_array, mask_image_path):
        return self.obtain_features(input_image_path, mask_image_path)

class No_Radiomics_Features(Radiomics_Features_Super_Class):

    def __init__(self, **kwargs):
        pass

    def obtain_features(self, input_image_path, mask_image_path):
        return []

class Radiomics_Features(Radiomics_Features_Super_Class):

    def __init__(self, **kwargs):
        pass

    def obtain_features(self, input_image_path, mask_image_path):
        features_to_return = []
        extractor = featureextractor.RadiomicsFeatureExtractor()

        input_image = sitk.ReadImage(input_image_path, sitk.sitkInt8)
        input_image_np = sitk.GetArrayViewFromImage(input_image)
        input_image_np = \
            cv2.resize(input_image_np.astype(np.float), dsize=(256, 256), \
                                                interpolation=cv2.INTER_CUBIC)
        input_image = sitk.GetImageFromArray(input_image_np.astype(np.uint8))

        mask_image = sitk.ReadImage(mask_image_path, sitk.sitkInt8)

        feature_extraction_result = extractor.execute(input_image, mask_image)

        headers_list = []
        for key, value in six.iteritems(feature_extraction_result):
            if (key.find('diagnostics')==-1):
                features_to_return.append(float(value))
                headers_list.append(key.replace('original_', ''))

        self.headers_list = headers_list

        return features_to_return
