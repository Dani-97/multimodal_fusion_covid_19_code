from imblearn.over_sampling import SMOTENC
import numpy as np

# This acts as an abstract class.
class Super_Balancing():

    def __init__(self):
        pass

    def execute_balancing(self, input_data, output_data, attrs_types_tuple):
        raise NotImplementedError("++++ ERROR: The execute_balancing method has not been implemented!")

class No_Balancing(Super_Balancing):

    def __init__(self):
        pass

    # As this class does not apply any kind of preprocessing, this method
    # will return the input_data without any kind of modification.
    # NOTE: attrs_types_tuple can be an empty tuple.
    def execute_balancing(self, input_data, output_data, attrs_types_tuple):
        print('**** No balancing will be applied to the training set')

        return input_data, output_data

class Oversampling_Balancing(Super_Balancing):

    def __init__(self):
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

    # NOTE: attrs_types_tuple can be an empty tuple.
    def execute_balancing(self, input_data, output_data, attrs_types_tuple):
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

    def __init__(self):
        pass

    def execute_balancing(self, input_data, output_data, attrs_types_tuple):
        print('**** The training subset is being oversampled with SMOTE...')

        top_features_categorical_attrs_headers, top_features_continuous_attrs_headers = attrs_types_tuple

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
