from imblearn.over_sampling import SMOTENC
import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

# This acts as an abstract class.
class Super_Balancing():

    def __init__(self, **kwargs):
        pass

    def execute_balancing(self, input_data, output_data, top_features_tuple):
        raise NotImplementedError("++++ ERROR: The execute_balancing method has not been implemented!")

class No_Balancing(Super_Balancing):

    def __init__(self, **kwargs):
        pass

    # As this class does not apply any kind of preprocessing, this method
    # will return the input_data without any kind of modification.
    # NOTE: top_features_tuple can be an empty tuple.
    def execute_balancing(self, input_data, output_data, top_features_tuple):
        print('**** No balancing will be applied to the training set')

        return input_data, output_data

class Oversampling_Balancing(Super_Balancing):

    def __init__(self, **kwargs):
        pass

    # nofsamples_of_the_class: Number of samples that belong to the current class.
    # nofsamples_to_add: Number of samples that must be oversampled on the current
    # class.
    def __generate_random_indexes(self, nofsamples_of_the_class, nofsamples_to_add):
        list_of_random_indexes = []

        if (nofsamples_to_add>0):
            for it in range(0, nofsamples_to_add):
                list_of_random_indexes.append(np.random.randint(0, nofsamples_of_the_class))

        return list_of_random_indexes

    # NOTE: top_features_tuple can be an empty tuple.
    def execute_balancing(self, input_data, output_data, top_features_tuple):
        print('**** The training subset is being oversampled...')

        output_data_shape = np.shape(output_data)

        samples_class0_input_data = input_data[output_data=='0', :]
        samples_class1_input_data = input_data[output_data=='1', :]

        samples_class0_output_data = output_data[output_data=='0']
        samples_class1_output_data = output_data[output_data=='1']

        nofsamples_class0 = np.shape(samples_class0_input_data)[0]
        nofsamples_class1 = np.shape(samples_class1_input_data)[0]
        max_nofsamples = np.max(np.array([nofsamples_class0, nofsamples_class1]))

        nof_samples_to_add_class0 = max_nofsamples-nofsamples_class0
        nof_samples_to_add_class1 = max_nofsamples-nofsamples_class1

        indexes_of_samples_to_add_class_0 = \
            self.__generate_random_indexes(nofsamples_class0, nof_samples_to_add_class0)
        indexes_of_samples_to_add_class_1 = \
            self.__generate_random_indexes(nofsamples_class1, nof_samples_to_add_class1)

        oversampled_input_data = \
            np.concatenate((samples_class0_input_data, \
                            samples_class0_input_data[indexes_of_samples_to_add_class_0, :], \
                            samples_class1_input_data, \
                            samples_class1_input_data[indexes_of_samples_to_add_class_1, :]), \
                            axis=0)
        oversampled_output_data = \
            np.concatenate((samples_class0_output_data, \
                            samples_class0_output_data[indexes_of_samples_to_add_class_0], \
                            samples_class1_output_data, \
                            samples_class1_output_data[indexes_of_samples_to_add_class_1]), \
                            axis=0)

        print('')
        print('**** INFO: With the oversampling:')
        print('Class 0 -> %d samples; Class 1 -> %d samples'\
                        %(np.sum(oversampled_output_data=='0'), \
                            np.sum(oversampled_output_data=='1')))
        print('')

        return oversampled_input_data, oversampled_output_data

class SMOTE_Balancing(Super_Balancing):

    def __init__(self, **kwargs):
        pass

    def execute_balancing(self, input_data, output_data, top_features_tuple):
        print('**** The training subset is being oversampled with SMOTE...')

        top_features_categorical_attrs_headers, top_features_continuous_attrs_headers, _ = top_features_tuple

        oversampling_obj = SMOTENC(top_features_categorical_attrs_headers)
        transformed_data = oversampling_obj.fit_resample(input_data, output_data)
        transformed_input_data, transformed_output_data = transformed_data

        print('')
        print('**** INFO: With the oversampling of SMOTE:')
        print('Class 0 -> %d samples; Class 1 -> %d samples'\
                        %(np.sum(transformed_output_data=='0'), \
                            np.sum(transformed_output_data=='1')))
        print('')


        return transformed_input_data, transformed_output_data

class Gaussian_Copula_Balancing(Super_Balancing):

    def __init__(self, **kwargs):
        self.manual_seed = kwargs['manual_seed']

    def execute_balancing(self, input_data, output_data, top_features_tuple):
        print('**** The training subset is being oversampled with Gaussian Copula Synthesizer...')

        _, _, top_features_attrs_headers = top_features_tuple
        input_df = pd.DataFrame(input_data)
        input_df.columns = top_features_attrs_headers
        only_output_column_df = pd.DataFrame(output_data)
        only_output_column_df.columns = ['output']

        input_df = pd.concat([input_df, only_output_column_df], axis=1)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=input_df)

        synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution='norm')

        nofsamples_list = np.array([len(input_df.query("output=='0'")), len(input_df.query("output=='1'"))])
        majority_class = np.argmax(nofsamples_list)
        minority_class = np.argmin(nofsamples_list)
        nofsamples_gap = nofsamples_list[majority_class] - nofsamples_list[minority_class]

        df_to_oversample = input_df.query("output=='%d'"%minority_class).replace(-1.0, np.nan)
        synthesizer.fit(df_to_oversample)

        synthetic_data = synthesizer.sample(num_rows=nofsamples_gap).replace(np.nan, -1.0)

        oversampled_data = pd.concat([input_df, synthetic_data])
        oversampled_data = oversampled_data.sample(frac=1, random_state=self.manual_seed)

        nof_samples_class0 = len(oversampled_data.query("output=='0'"))
        nof_samples_class1 = len(oversampled_data.query("output=='1'"))
        print('')
        print('**** INFO: With the oversampling of Gaussian Copula:')
        print('Class 0 -> %d samples; Class 1 -> %d samples'%(nof_samples_class0, nof_samples_class1))
        print('')

        return oversampled_data[oversampled_data.columns[:-1]].values, oversampled_data['output'].values
