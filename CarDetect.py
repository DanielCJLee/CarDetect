"""
Created on Jul 3, 2013

@author: Zachary
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import average
from scipy.io import wavfile
from datetime import datetime

EPSILON = np.finfo(np.double).eps

FRAME_TIME_LENGTH = 50 # length of frame in milliseconds
DIVISIONS = np.array([40, 70, 110, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 11025])
#DIVISIONS = np.array([1000,1500,2000,2500,3000,3500,4000,5000,7000,10000])
MOVING_AVERAGE_LENGTH = 1000 / FRAME_TIME_LENGTH # length in number of FFTs


def nextpow2(num):
    return int(np.ceil(np.log2(num)))


def spect_plot(bins, freqs, slices, logscale=True, axes=plt):
    power = slices.T
    if logscale:
        z = np.log10(power)
    else:
        z = power
    axes.pcolormesh(bins, freqs, z)


def find_indexes(freqs, divisions):
    # Determine where the divisions are in the freqs list

    indexes = []
    i = 0
    for div in divisions:
        while i < len(freqs) and freqs[i] < div:
            i += 1
        indexes.append(i)

    return indexes


def list_sum(list_of_matrices):
    total = list_of_matrices[0]
    for i in xrange(1, len(list_of_matrices)):
        total = [sum(pair) for pair in zip(total, list_of_matrices[i])]
    return total


def freq_bins(freqs, slices, divisions):
    # Divide slices into frequency bins, returns new slices

    indexes = find_indexes(freqs, divisions)

    power = slices.T
    output = []

    prev_index = 0
    for index in indexes:
        output.append(sum(power[prev_index:index + 1]))
        prev_index = index

    output = np.array(output)

    return output


def moving_average(slices):
    averages = []
    for end in xrange(len(slices)):
        start = max(0, end - MOVING_AVERAGE_LENGTH - 1)
        actual_length = end - start + 1
        average = sum(slices[start:end + 1]) / actual_length
        # note there is some imprecision in using integer instead of float math
        # but this is also faster, and easier to implement
        averages.append(average)
    averages = np.array(averages)
    return averages


def trim_outliers(data, num_std_devs=3):
    data10 = np.log10(data)

    sd10 = np.std(data10)
    mean10 = np.average(data10)

    #output = min(data, mean10+num_std_devs*sd10)
    #output = max(output, mean10-num_std_devs*sd10)

    output = np.copy(data)

    lower_bound10 = mean10 - num_std_devs * sd10
    upper_bound10 = mean10 + num_std_devs * sd10

    lower_bound = 10 ** lower_bound10
    upper_bound = 10 ** upper_bound10

    print upper_bound10, lower_bound10

    num_high = 0
    num_low = 0
    count = 0

    for elem in np.nditer(output, op_flags=['readwrite']):
        elem10 = np.log10(elem)
        if elem10 > upper_bound10:
            elem[...] = upper_bound
            num_high += 1
        elif elem10 < lower_bound10:
            elem[...] = lower_bound
            num_low += 1
        count += 1

    print "# high", num_high
    print "# low", num_low
    print "# total", count
    return output


def rolloff_freq(slice, threshold=0.90):
    target = threshold * sum(slice)
    partial = 0.0
    i = 0
    length = len(slice)
    while partial < target and i < length - 1:
        partial += slice[i]
        i += 1
    return i


def graph_rolloff_freq(freqs, slices):
    return [freqs[rolloff_freq(x)] for x in slices]


def avg_zero_crossing_rate(sound_data):
    signs = np.sign(np.array(sound_data))
    total = 0
    for i in xrange(1, len(signs)):
        if signs[i - 1] != signs[i]:
            total += 1
    rate = float(total) / len(sound_data)
    return rate


def normalize(slices):
    output = []
    for slice in slices:
        output.append(slice / np.average(slice))
    return np.array(output).T


def main():
    rate, data = wavfile.read('./recordings/carNight1.wav')
    print "Sound file loaded"

    framelen_samples = int(float(FRAME_TIME_LENGTH) / float(1000) * float(rate))
    noverlap = int(0.3 * framelen_samples)
    NFFT = 2 ** nextpow2(framelen_samples)

    (power, freqs, bins, im) = plt.specgram(x=data, NFFT=NFFT, Fs=rate, noverlap=noverlap)
    plt.cla()
    print "Computed spectrogram"

    slices = power.T

    # Divide into useful frequency bins
    #p3 = freq_bins(freqs, slices, divisions)
    #freqs = divisions

    # Find differences from the moving average (filters out some background noise)
    differences = abs(slices - moving_average(slices))
    differences[differences == 0] = EPSILON # replace zero values with small number

    differences = trim_outliers(differences, num_std_devs=3)

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    spect_plot(bins, freqs, differences, logscale=True, axes=ax1)
    #plt.show()

    # Plot zero-crossing rate in intervals of the FFT window
    sample_size = int((float(FRAME_TIME_LENGTH) / 1000) * rate)
    x = []
    y = []
    num = int(len(data) / sample_size)
    for i in xrange(num):
        x.append((i + 1) * FRAME_TIME_LENGTH)
        y.append(avg_zero_crossing_rate(data[i * sample_size:(i + 1) * sample_size]))
    ax2.plot(x, y)

    # Plot rolloff frequency
    ax3.plot(bins, graph_rolloff_freq(freqs, slices))

    plt.show()


if __name__ == '__main__':
    main()