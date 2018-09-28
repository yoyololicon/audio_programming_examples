from ch9.utils import *
from ch8.stft import rfft, signal, irfft
import argparse
from librosa import load
from librosa.output import write_wav

parser = argparse.ArgumentParser()
parser.add_argument('infile', help='input filename')
parser.add_argument('outfile', help='output filename')
parser.add_argument('--dur', default=10, help='duration of input in seconds')


def pva(input, fftsize, hopsize, sr, window=None):
    if type(window) == str:
        window = signal.get_window(window, fftsize)
    else:
        window = np.ones(fftsize)

    fac = sr / (hopsize * 2 * np.pi)
    scal = 2 * np.pi * hopsize / fftsize

    a = (len(input) - fftsize) % hopsize
    if a:
        input = np.pad(input, (0, hopsize - a), 'constant', constant_values=0)

    rotated_frames = np.stack(
        [np.roll(input[i:i + fftsize] * window, shift=i) for i in range(0, len(input) - fftsize + 1, hopsize)], axis=1)

    spec = rfft(rotated_frames, 0)

    # polar form
    deltaphi(spec)
    k = np.arange(1, fftsize // 2)
    spec[3::2] += k[:, None] * scal
    spec[3::2] *= fac

    return spec


def pvs(input, hopsize, sr, windowtype):
    fftsize, t = input.shape
    if type(windowtype) == str:
        window = signal.get_window(windowtype, fftsize)
    else:
        window = np.ones(fftsize)

    fac = hopsize * np.pi * 2 / sr
    scal = sr / fftsize
    k = np.arange(1, fftsize // 2)

    input[3::2] -= k[:, None] * scal
    input[3::2] *= fac

    sigmaphi(input)

    frames = irfft(input, 0)

    output = np.zeros(hopsize * t + fftsize - hopsize)
    for i in range(t):
        pos = i * hopsize
        output[pos:pos + fftsize] += np.roll(frames[:, i], shift=-pos) * window
    return output


def main(infile, outfile, dur):
    fftsize = 1024
    hopsize = 256

    data, sr = load(infile, sr=None, duration=dur)

    pv = pva(data, fftsize, hopsize, sr, 'hanning')
    y = pvs(pv, hopsize, sr, 'hanning')

    write_wav(outfile, y, sr)


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.infile, args.outfile, int(args.dur))
