import argparse
import pandas as pd

class Analysis_Only_Hospitalized():

    def __init__(self):
        pass

    # This function uses the name of the attribute as the input as well as the
    # input dataframe.
    def __perform_individual_analysis__(self, attr, input_dataframe):
        # Total number of rows of the dataset.
        nof_rows = len(input_dataframe)

        nof_attr = len(input_dataframe.query('%s==1'%attr))
        nof_non_attr = len(input_dataframe.query('%s!=1'%attr))

        nof_attr_and_class0 = len(input_dataframe.query('(%s==1) and (output==0)'%attr))
        nof_attr_and_class1 = len(input_dataframe.query('(%s==1) and (output==1)'%attr))

        nof_non_attr_and_class0 = len(input_dataframe.query('(%s!=1) and (output==0)'%attr))
        nof_non_attr_and_class1 = len(input_dataframe.query('(%s!=1) and (output==1)'%attr))

        print('##################### STATISTICS OF %s ##########################'%attr)

        percentage_aux = (nof_attr/nof_rows)*100
        print('++++ Number of rows with %s set to Positive = [%d (%.2f %s)]'%(attr, nof_attr, percentage_aux, '%'))
        percentage_aux = (nof_non_attr/nof_rows)*100
        print('++++ Number of rows with %s set to Negative = [%d (%.2f %s)]'%(attr, nof_non_attr, percentage_aux, '%'))

        percentage_aux = (nof_attr_and_class0/nof_attr)*100
        print('++++ Number of samples with %s Positive and Survival = [%d (%.2f %s)]'%(attr, \
                                              nof_attr_and_class0, percentage_aux, '%'))

        percentage_aux = (nof_attr_and_class1/nof_attr)*100
        print('++++ Number of samples with %s Positive and Death = [%d (%.2f %s)]'%(attr, \
                                                nof_attr_and_class1, percentage_aux, '%'))

        percentage_aux = (nof_non_attr_and_class0/nof_non_attr)*100
        print('++++ Number of samples with %s Negative and Survival = [%d (%.2f %s)]'%(attr, \
                                                nof_non_attr_and_class0, percentage_aux, '%'))

        percentage_aux = (nof_non_attr_and_class1/nof_non_attr)*100
        print('++++ Number of samples with %s Negative and Death = [%d (%.2f %s)]'%(attr, \
                                                nof_non_attr_and_class1, percentage_aux, '%'))

        print('')

    def execute_analysis(self, input_dataframe):
        attrs_list = ['hta', 'diabetes_mellitus', 'epoc', 'asma', 'hepatopatia', 'leucemia', \
            'linfoma', 'neoplasia', 'hiv', 'transplante_organo_solido', \
                'quimioterapia_ultimos_3_meses', 'biologicos_ultimos_3_meses', \
                    'corticoides_cronicos_mas_3_meses']

        for attr_aux in attrs_list:
            self.__perform_individual_analysis__(attr_aux, input_dataframe)
