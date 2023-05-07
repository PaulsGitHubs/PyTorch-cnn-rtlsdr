import time
import torch
import numpy as np
import os
import sys, argparse
from rtlsdr import RtlSdr
import scipy.signal as signal


def read_samples(sdr, freq):
    f_offset = 250000  # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ppm', type=int, default=0,
                    help='dongle ppm error correction')
parser.add_argument('--gain', type=int, default=20,
                    help='dongle gain level')
parser.add_argument('--threshold', type=float, default=0.75,
                    help='threshold to display/hide predictions')
parser.add_argument('--start', type=int, default=85000000,
                    help='begin scan here, in Hertz')
parser.add_argument('--stop', type=int, default=108000000,
                    help='stop scan here, in Hertz')
parser.add_argument('--step', type=int, default=100000,
                    help='step size for scan, in Hertz')

args = parser.parse_args()

sdr = RtlSdr()
sdr.sample_rate = sample_rate = 2400000
sdr.err_ppm = args.ppm
sdr.gain = args.gain

classes = [d for d in os.listdir('training_data') if os.path.isdir(os.path.join('training_data', d))]
num_classes = len(classes)

model = torch.load('rtlsdr-model.pt')
model.eval()

y_test_samples = np.zeros((1, num_classes))

freq = args.start
while freq <= args.stop:
    samples = []

    iq_samples = read_samples(sdr, freq)
    iq_samples = signal.decimate(iq_samples, 48)

    real = np.real(iq_samples)
    imag = np.imag(iq_samples)

    iq_samples = []
    for i in range(0, np.ma.count(real) - 212):  # 128*192 magic
        iq_samples.append(real[i])
        iq_samples.append(imag[i])
    iq_samples = np.reshape(iq_samples, (-1, 128, 2))

    samples.append(iq_samples)

    samples = np.array(samples)

    x_batch = torch.tensor(samples.reshape(1, 2, 96, 128)).float()

    with torch.no_grad():
        result = model(x_batch)

    max = 0.0
    maxlabel = ""
    for sigtype, probability in zip(classes, result[0]):
        if probability >= max:
            max = probability
            maxlabel = sigtype

    if (maxlabel != 'other') and (max >= args.threshold):
        print('{:.3f} MHz - {} {:.2f}%'.format(freq / 1e6, maxlabel, max * 100))

    freq += args.step

sdr.close()
