# just plotting stuff for debug
import numpy as np
import matplotlib.pyplot as plt


class PlottingUtils:
    plt.figure()
    plt.ion()
    def closeAllPlots(self):
        plt.close('all')
    def clearfig(self):
        plt.clf()
    # x-axis: 0, y-axis: 1
    def energyByAxis(self, img, axis = 0):
        first_channel = img[:,:,0]
        total_energy = np.sum(first_channel,axis=axis)
        
        title_axis = ["by column","by row"]
        plt.plot(total_energy)
        plt.legend(title_axis)
        plt.draw()
        plt.pause(0.01)
        return total_energy
