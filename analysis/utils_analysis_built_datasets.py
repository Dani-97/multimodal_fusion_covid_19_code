import argparse
import pandas as pd

class Super_Class_Analysis():

    def __init__(self):
        pass

    def __perform_individual_analysis_markdown__(self, attr, input_dataframe, name_class0, name_class1):
        # Total number of rows of the dataset.
        nof_rows = len(input_dataframe)

        nof_attr = len(input_dataframe.query('%s==1'%attr))
        nof_non_attr = len(input_dataframe.query('%s!=1'%attr))

        nof_attr_and_class0 = len(input_dataframe.query('(%s==1) and (output==0)'%attr))
        nof_attr_and_class1 = len(input_dataframe.query('(%s==1) and (output==1)'%attr))

        nof_non_attr_and_class0 = len(input_dataframe.query('(%s!=1) and (output==0)'%attr))
        nof_non_attr_and_class1 = len(input_dataframe.query('(%s!=1) and (output==1)'%attr))

        print('| **%s** |                  |                  |'%attr)
        print('|:-------------------:|:----------------:|:----------------:|')
        percentage_si_aux = (nof_attr/nof_rows)*100
        percentage_no_aux = (nof_non_attr/nof_rows)*100
        print('|                     | **No** (%.2f %s) | **Sí** (%.2f %s) |'%\
                                        (percentage_no_aux, '%', percentage_si_aux, '%'))
        percentage_class0_no_attr_aux = (nof_non_attr_and_class0/nof_non_attr)*100
        percentage_class0_si_attr_aux = (nof_attr_and_class0/nof_attr)*100
        print('|      **%s**     |      %.2f %s    |      %.2f %s     |'%\
                (name_class0, percentage_class0_no_attr_aux, '%', \
                              percentage_class0_si_attr_aux, '%'))
        percentage_class1_no_attr_aux = (nof_non_attr_and_class1/nof_non_attr)*100
        percentage_class1_si_attr_aux = (nof_attr_and_class1/nof_attr)*100
        print('|      **%s**     |      %.2f %s    |      %.2f %s     |'%\
                (name_class1, percentage_class1_no_attr_aux, '%', \
                              percentage_class1_si_attr_aux, '%'))
        print('')

    # This function uses the name of the attribute as the input as well as the
    # input dataframe.
    def __perform_individual_analysis__(self, attr, input_dataframe, name_class0, name_class1):
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
        print('++++ Number of samples with %s Positive and %s = [%d (%.2f %s)]'%(attr, \
                                    name_class0, nof_attr_and_class0, percentage_aux, '%'))

        percentage_aux = (nof_attr_and_class1/nof_attr)*100
        print('++++ Number of samples with %s Positive and %s = [%d (%.2f %s)]'%(attr, \
                                    name_class1, nof_attr_and_class1, percentage_aux, '%'))

        percentage_aux = (nof_non_attr_and_class0/nof_non_attr)*100
        print('++++ Number of samples with %s Negative and %s = [%d (%.2f %s)]'%(attr, \
                                    name_class0, nof_non_attr_and_class0, percentage_aux, '%'))

        percentage_aux = (nof_non_attr_and_class1/nof_non_attr)*100
        print('++++ Number of samples with %s Negative and %s = [%d (%.2f %s)]'%(attr, \
                                    name_class1, nof_non_attr_and_class1, percentage_aux, '%'))

        print('')

class Analysis_Only_Hospitalized(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe):
        attrs_list = ['hta', 'diabetes_mellitus', 'epoc', 'asma', 'hepatopatia', 'leucemia', \
            'linfoma', 'neoplasia', 'hiv', 'transplante_organo_solido', \
                'quimioterapia_ultimos_3_meses', 'biologicos_ultimos_3_meses', \
                    'corticoides_cronicos_mas_3_meses']
        name_class0, name_class1 = 'Non Exitus', 'Exitus'

        for attr_aux in attrs_list:
            super().__perform_individual_analysis_markdown__(attr_aux, input_dataframe, name_class0, name_class1)

class Analysis_Only_Hospitalized_Joint_Inmunosupression(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe):
        attrs_list = ['inmunosupression']
        name_class0, name_class1 = 'Non Exitus', 'Exitus'

        for attr_aux in attrs_list:
            super().__perform_individual_analysis_markdown__(attr_aux, input_dataframe, name_class0, name_class1)

class Analysis_Hospitalized_And_Urgencies(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe):
        attrs_list = ['hta', 'diabetes_mellitus', 'epoc', 'asma', 'hepatopatia', 'leucemia', \
            'linfoma', 'neoplasia', 'hiv', 'transplante_organo_solido', \
                'quimioterapia_ultimos_3_meses', 'biologicos_ultimos_3_meses', \
                    'corticoides_cronicos_mas_3_meses']
        name_class0, name_class1 = 'Hospitalized', 'Urgencies'

        for attr_aux in attrs_list:
            super().__perform_individual_analysis_markdown__(attr_aux, input_dataframe, name_class0, name_class1)
