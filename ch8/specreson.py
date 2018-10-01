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

    pi2sr = 2 * np.pi / sr
    durf = spec.shape[1]

    cf = np.linspace(100, 3000, durf)
    bw = 100
    radius = 1 - pi2sr * bw / 2
    radsq = radius ** 2
    angle = cf * pi2sr * 2 * radius / (1 + radsq)
    scale = (1 - radsq) * np.sin(angle) * cf / (4 * bw)

    for i in range(durf):
        spec[:, i] = specreson(spec[:, i], scale[i], angle[i], radius)
    output = istft(spec, hopsize, 'hanning')
    write_wav(outfile, output, sr)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.infile, args.outfile)
