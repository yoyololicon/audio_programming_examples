from ch8.stft import *
import argparse
from librosa import load
from librosa.output import write_wav

parser = argparse.ArgumentParser()
parser.add_argument('infile1', help='input1 (magnitudes) filename')
parser.add_argument('infile2', help='input2 (phases) filename')
parser.add_argument('outfile', help='output filename')
parser.add_argument('dur', default=10, help='duration of input in seconds')


def crosspec(maginput, phasinput):
    output = maginput.copy()
    mag = np.sqrt(np.power(maginput[2::2], 2) + np.power(maginput[3::2], 2))
    phi = np.arctan2(phasinput[3::2], phasinput[2::2])
    output[2::2] = mag * np.cos(phi)
    output[3::2] = mag * np.sin(phi)
    return output


def main(infile1, infile2, outfile, dur):
    fftsize = 1024
    hopsize = 256

    data1, sr1 = load(infile1, sr=None, duration=dur)
    data2, sr2 = load(infile2, sr=None, duration=dur)
    if sr1 != sr2:
        print("sampling rate not equal.")
        exit(1)

    spec1 = stft(data1, fftsize, hopsize, 'hanning')
    spec2 = stft(data2, fftsize, hopsize, 'hanning')

    maxlen = min(spec1.shape[1], spec2.shape[1])
    spec1, spec2 = spec1[:, :maxlen], spec2[:, :maxlen]
    final_spec = crosspec(spec1, spec2)

    output = istft(final_spec, hopsize, 'hanning')
    write_wav(outfile, output, sr1)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.infile1, args.infile2, args.outfile, int(args.dur))
