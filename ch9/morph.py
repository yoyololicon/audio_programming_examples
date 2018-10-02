import argparse
import numpy as np
from librosa import load
from librosa.output import write_wav
from ch9.utils import pvmorph
from ch9.pv import pva, pvs

parser = argparse.ArgumentParser()
parser.add_argument('infile1', help='input1 (start sound) filename')
parser.add_argument('infile2', help='input2 (final sounds) filename')
parser.add_argument('outfile', help='output filename')
parser.add_argument('--dur', default=10, help='duration of input in seconds')
parser.add_argument('--bkp', help='break points file')


def main(infile1, infile2, outfile, dur, brkfile):
    fftsize = 1024
    hopsize = 256

    data1, sr1 = load(infile1, sr=None, duration=dur)
    data2, sr2 = load(infile2, sr=None, duration=dur)
    if sr1 != sr2:
        print("sampling rate not equal.")
        exit(1)

    pv1 = pva(data1, fftsize, hopsize, sr1, 'hanning')
    pv2 = pva(data2, fftsize, hopsize, sr2, 'hanning')

    maxlen = min(pv1.shape[1], pv2.shape[1])
    pv1, pv2 = pv1[:, :maxlen], pv2[:, :maxlen]
    t = np.arange(maxlen) * hopsize / sr1

    if brkfile:
        x = np.loadtxt(brkfile, delimiter='\t', skiprows=1)
        if x[-1, 0] < t[-1]:
            x = np.concatenate((x, np.array([[t[-1], 0, 0]])), axis=0)
        if x[0, 0] > 0:
            x = np.concatenate((np.zeros((1,3)), x), axis=0)
        tbrk, morpha, morphfr = x[:, 0], x[:, 1], x[:, 2]
        morpha = np.interp(t, tbrk, morpha)
        morphfr = np.interp(t, tbrk, morphfr)
    else:
        morpha = morphfr = np.linspace(0, 1, maxlen)

    morphed = np.stack([pvmorph(pv1[:, i], pv2[:, i], morpha[i], morphfr[i]) for i in range(maxlen)], 1)

    output = pvs(morphed, hopsize, sr1, 'hanning')
    write_wav(outfile, output, sr1)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.infile1, args.infile2, args.outfile, int(args.dur), args.bkp)
