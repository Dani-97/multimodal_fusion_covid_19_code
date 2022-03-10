import numpy as np
import matplotlib.pyplot as plt
from utils import convert_emtpy_str_to_integer, convert_emtpy_str_to_float
from utils import compute_number_of_missing_values_empty_str, detect_missing_values

# This class is the parent of all the classes that are created to
# obtain the histogram of each attribute that want to be studied.
class SuperClass_Obtain_Histogram():

    # nofattribute = this parameter chooses the number of attribute that want
    # to be analyzed.
    # title = title of the plot.
    # nofbins = number of bins used on the histogram.
    def __init__(self, nofattribute, title, nofbins):
        self.nofattribute = nofattribute
        self.title = title
        self.nofbins = nofbins
        # This variable will label only the number of missing values,
        # hiding the labels for the rest of the x values.
        self.hide_labels = False
        self.y_log_scale = False
        self.x_axis_visible = True

    def set_hide_labels(self, value):
        self.hide_labels = value

    def set_log_scale(self, value):
        self.y_log_scale = value

    def set_x_axis_visibility(self, value):
        self.x_axis_visible = value

    def plot_histogram(self, file_data_for_histogram, output_filename):
        fig, ax = plt.subplots()

        bins = self.nofbins
        converted_data_for_histogram = list(map(detect_missing_values, file_data_for_histogram[self.nofattribute]))
        plt.hist(converted_data_for_histogram, bins=bins, rwidth=0.75, edgecolor = 'black')

        rects = ax.patches
        xticks = []
        it = 0
        for rect_aux in rects:
            height = rect_aux.get_height()
            if (not self.hide_labels):
                ax.text(rect_aux.get_x() + rect_aux.get_width() / 2, height+0.01, '%d'%height,
                                    ha='center', va='bottom')
            xticks.append(rect_aux.get_x() + rect_aux.get_width() / 2)
            it+=1

        if (not self.x_axis_visible):
            plt.xticks([])
        else:
            plt.xticks(xticks)

        if (self.y_log_scale):
            plt.yscale('log')

        plt.title(self.title + ' | Missing values = %d'% \
            compute_number_of_missing_values_empty_str(file_data_for_histogram[self.nofattribute]))
        plt.savefig(output_filename)
        plt.close()

class Obtain_Histogram_Attr_0(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_1(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_2(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_3(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_4(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)

class Obtain_Histogram_Attr_5(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)

class Obtain_Histogram_Attr_6(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)

class Obtain_Histogram_Attr_7(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)

class Obtain_Histogram_Attr_8(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_X(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

    # In this case, we want to override the original method because we need
    # some ad-hoc procedures.
    def plot_histogram(self, file_data_for_histogram, output_filename):
        fig, ax = plt.subplots()

        bins = self.nofbins
        # We need to convert the empty strings to -1 in order to avoid
        # conversion problems. This is stored in the variable declared below.
        converted_data_for_histogram = list(map(convert_emtpy_str_to_integer, file_data_for_histogram[11]))
        plt.hist(np.sort(converted_data_for_histogram), bins=bins, rwidth=0.75,  \
                                                               edgecolor = 'black')

        rects = ax.patches
        it = 0
        for rect_aux in rects:
            height = rect_aux.get_height()
            if (it==0):
                ax.text(rect_aux.get_x() + rect_aux.get_width() / 2, height+0.01, 'missing_values = %d'%height,
                        ha='center', va='bottom')
                it+=1

        plt.yscale('log')
        plt.title(self.title)
        plt.savefig(output_filename)
        plt.close()

class Obtain_Histogram_Attr_X(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

    # In this case, we want to override the original method because we need
    # some ad-hoc procedures.
    def plot_histogram(self, file_data_for_histogram, output_filename):
        fig, ax = plt.subplots()

        bins = self.nofbins
        change_commas_to_dots = lambda input: input.replace(',', '.')
        converted_data_for_histogram = list(map(change_commas_to_dots, file_data_for_histogram[12]))
        # We need to convert the empty strings to -1 in order to avoid
        # conversion problems. This is stored in the variable declared below.
        converted_data_for_histogram = list(map(convert_emtpy_str_to_float, converted_data_for_histogram))
        plt.hist(np.sort(converted_data_for_histogram), bins=bins, rwidth=0.75,  \
                                                               edgecolor = 'black')

        rects = ax.patches
        it=0
        for rect_aux in rects:
            height = rect_aux.get_height()
            if (it==0):
                ax.text(rect_aux.get_x() + rect_aux.get_width() / 2, height+0.01, 'missing_values = %d'%height,
                        ha='center', va='bottom')
                it+=1

        plt.yscale('log')
        plt.title(self.title)
        plt.savefig(output_filename)
        plt.close()

class Obtain_Histogram_Attr_X(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

    # In this case, we want to override the original method because we need
    # some ad-hoc procedures.
    def plot_histogram(self, file_data_for_histogram, output_filename):
        fig, ax = plt.subplots()

        bins = self.nofbins
        change_commas_to_dots = lambda input: input.replace(',', '.')
        converted_data_for_histogram = list(map(change_commas_to_dots, file_data_for_histogram[13]))
        # We need to convert the empty strings to -1 in order to avoid
        # conversion problems. This is stored in the variable declared below.
        converted_data_for_histogram = list(map(convert_emtpy_str_to_float, converted_data_for_histogram))
        plt.hist(np.sort(converted_data_for_histogram), bins=bins, rwidth=0.75,  \
                                                               edgecolor = 'black')

        rects = ax.patches
        it = 0
        for rect_aux in rects:
            height = rect_aux.get_height()
            if (it==0):
                ax.text(rect_aux.get_x() + rect_aux.get_width() / 2, height+0.01, 'missing_values = %d'%height,
                        ha='center', va='bottom')
                it+=1

        plt.yscale('log')
        plt.title(self.title)
        plt.savefig(output_filename)
        plt.close()

class Obtain_Histogram_Attr_9(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_10(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_11(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_12(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_13(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_14(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_15(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_16(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_17(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_18(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_19(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_20(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_21(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_22(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)

class Obtain_Histogram_Attr_23(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)
        super().set_log_scale(True)

class Obtain_Histogram_Attr_24(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)
        super().set_log_scale(True)

class Obtain_Histogram_Attr_25(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)
        super().set_log_scale(True)

class Obtain_Histogram_Attr_26(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)
        super().set_log_scale(True)

class Obtain_Histogram_Attr_27(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)
        super().set_log_scale(True)

class Obtain_Histogram_Attr_28(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)
        super().set_log_scale(True)

class Obtain_Histogram_Attr_29(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)
        super().set_log_scale(True)

class Obtain_Histogram_Attr_30(SuperClass_Obtain_Histogram):

    def __init__(self, nofattribute, title, nofbins):
        super().__init__(nofattribute, title, nofbins)
        super().set_hide_labels(True)
        super().set_x_axis_visibility(False)
        super().set_log_scale(True)
