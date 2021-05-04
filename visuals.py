"""
visuals.py to be used to visualize node voltage matrices with color uwu

use this for reference:
https://github.com/teadetime/softdesFinal/blob/master/bottom_up.py
"""

import numpy as np
import matplotlib.pyplot as plt

class Visual(object):
    def __init__(self, color_scheme = 'hot'):
        self.color_scheme = color_scheme

    def visualize_node_voltage(self, mat, Vdd=5):
        # generate heatmap based on
        # scale values from between and min to between 0 and 1
        if mat.min() < 0:
            # shift entire mat up
            mat += mat.min()

        mat *= 1.0/mat.max()

        plt.imshow(mat, cmap=self.color_scheme, interpolation='nearest')
        plt.title('Voltage Matrix heatmap')
        plt.show(block=True)

    def visulaize_current(self, mat):
        pass

    def visulaize_resistance(self, mat):
        pass

if __name__ == "__main__":
    pass
