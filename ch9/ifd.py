import argparse
import numpy as np
from librosa import load
from librosa.output import write_wav
from ch9.utils import ifd, addsyn

parser = argparse.ArgumentParser()
parser.add_argument('infile', help='input filename')
parser.add_argument('outfile', help='output filename')
parser.add_argument('--pitch', type=float, default=1)
parser.add_argument('--scale', type=float, default=0.004)
parser.add_argument('-t', action='store_true', help='transpose (reverse) the input')
parser.add_argument('--dur_ratio', type=float, default=1, help='time stretch control')


def main(infile, outfile, pitch, scale, transpose, dur_ratio):
    fftsize = 1024
    hopsize = 256

    data, sr = load(infile, sr=None)

    pv = ifd(data, fftsize, hopsize, sr, 'hanning')
    if transpose:
        pv = np.flip(pv, 1)
    output = addsyn(pv, 0, pitch, scale, int(hopsize * dur_ratio), sr)

    write_wav(outfile, output, sr)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.infile, args.outfile, args.pitch, args.scale, args.t, args.dur_ratio)
