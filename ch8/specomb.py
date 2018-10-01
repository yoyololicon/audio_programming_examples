import argparse
from librosa import load
from librosa.output import write_wav
from ch8.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('infile', help='input filename')
parser.add_argument('outfile', help='output filename')


def main(infile, outfile):
    fftsize = 1024
    hopsize = 256

    data, sr = load(infile, sr=None)

    spec = stft(data, fftsize, hopsize, 'hanning')

    fr = sr / hopsize
    twopi = np.pi * 2
    delay = 0.0055 + 5e-3 * np.cos(np.arange(spec.shape[1]) * twopi * 0.1 / fr)

    for i in range(spec.shape[1]):
        spec[:, i] = specomb(spec[:, i], 0.6, delay[i], 0.9, sr)
    output = istft(spec, hopsize, 'hanning')
    write_wav(outfile, output, sr)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.infile, args.outfile)
