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
    output = simplp(spec)
    output = istft(output, hopsize, 'hanning')
    write_wav(outfile, output, sr)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.infile, args.outfile)