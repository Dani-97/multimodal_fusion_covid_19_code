import numpy as np

class No_Preprocessing():

    def __init__(self):
        pass

    # As this class does not apply any kind of preprocessing, this method
    # will return the input_data without any kind of modification.
    def execute_preprocessing(self, input_data, output_data):
        print('**** No preprocessing will be applied to the dataset')

        return input_data, output_data

class Undersampling_Preprocessing():

    def __init__(self):
        pass

    def execute_preprocessing(self, input_data, output_data):
        output_data_shape = np.shape(output_data)

        samples_class0_input_data = input_data[output_data=='0', :]
        samples_class1_input_data = input_data[output_data=='1', :]

        samples_class0_output_data = output_data[output_data=='0']
        samples_class1_output_data = output_data[output_data=='1']

        nofsamples_class0 = np.shape(samples_class0_input_data)[0]
        nofsamples_class1 = np.shape(samples_class1_input_data)[0]
        min_nofsamples = np.min(np.array([nofsamples_class0, nofsamples_class1]))

        undersampled_input_data = \
            np.concatenate((samples_class0_input_data[0:min_nofsamples, :], \
                        samples_class1_input_data[0:min_nofsamples, :]), \
                            axis=0)
        undersampled_output_data = \
            np.concatenate((samples_class0_output_data[0:min_nofsamples], \
                        samples_class1_output_data[0:min_nofsamples]), \
                            axis=0)

        return undersampled_input_data, undersampled_output_data
