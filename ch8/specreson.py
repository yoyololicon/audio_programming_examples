import argparse
from librosa import load
from librosa.output import write_wav
from ch8.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('infile', help='input filename')
parser.add_argument('outfile', help='output filename')
parser.add_argument('--cf_file', help='central frequency breakpoints file')
parser.add_argument('--bw_file', help='bandwidth breakpoints file')


def main(infile, outfile, cf_file, bw_file):
    fftsize = 1024
    hopsize = 256

    data, sr = load(infile, sr=None)

    spec = stft(data, fftsize, hopsize, 'hanning')

    pi2sr = 2 * np.pi / sr
    durf = spec.shape[1]
    t = np.arange(durf) * hopsize / sr

    if cf_file:
        x = np.loadtxt(cf_file, delimiter='\t', skiprows=1)
        if x[-1, 0] < t[-1]:
            x = np.vstack((x, np.array([[t[-1], 0]])))
        if x[0, 0] > 0:
            x = np.vstack((np.zeros((1, 2)), x))
        tbrk, cf = x[:, 0], x[:, 1]
        cf = np.interp(t, tbrk, cf)
    else:
        cf = np.linspace(100, 3000, durf)

    if bw_file:
        x = np.loadtxt(bw_file, delimiter='\t', skiprows=1)
        if x[-1, 0] < t[-1]:
            x = np.vstack((x, np.array([[t[-1], 0]])))
        if x[0, 0] > 0:
            x = np.vstack((np.zeros((1, 2)), x))
        tbrk, bw = x[:, 0], x[:, 1]
        bw = np.interp(t, tbrk, bw)
    else:
        bw = np.full(durf, 100)

    radius = 1 - pi2sr * bw / 2
    radsq = radius ** 2
    angle = cf * pi2sr * 2 * radius / (1 + radsq)
    scale = (1 - radsq) * np.sin(angle) * cf / (4 * bw)

    for i in range(durf):
        spec[:, i] = specreson(spec[:, i], scale[i], angle[i], radius[i])
    output = istft(spec, hopsize, 'hanning')
    write_wav(outfile, output, sr)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.infile, args.outfile, args.cf_file, args.bw_file)
