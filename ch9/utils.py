import numpy as np
from scipy import signal, interpolate
from ch8.stft import rfft


def deltaphi(spec):
    mag = np.sqrt(spec[2::2] ** 2 + spec[3::2] ** 2)
    phi = np.arctan2(spec[3::2], spec[2::2])

    spec[2::2] = mag
    spec[3::2, 0] = phi[:, 0]
    phi_diff = np.diff(phi, axis=1)

    phi_diff = np.unwrap(phi_diff)
    spec[3::2, 1:] = phi_diff


def sigmaphi(spec):
    mag = spec[2::2].copy()
    phi_diff = spec[3::2]
    phi_sum = np.cumsum(phi_diff, axis=1)

    spec[2::2] = mag * np.cos(phi_sum)
    spec[3::2] = mag * np.sin(phi_sum)


def pvmorph(input1, input2, morpha, morphfr):
    output = np.empty(input1.shape, input1.dtype)
    output[::2] = input1[::2] + (input2[::2] - input1[::2]) * morpha
    # nyquist freq
    output[1] = input1[1] + (input2[1] - input1[1]) * morpha

    div = input2[3::2] / input1[3::2]
    div = np.abs(div)

    output[3::2] = input1[3::2] * np.power(div, morphfr)
    return output


def ifd(input, fftsize, hopsize, sr, window=None):
    padsize = 0
    if type(window) == str:
        winsize = fftsize
        window = signal.get_window(window, fftsize)
    elif type(window) == np.ndarray:
        winsize = len(window)
        padsize = fftsize - winsize
    else:
        winsize = fftsize
        window = np.ones(winsize)

    diffwin = -np.gradient(window)
    fac = sr / (2 * np.pi)
    fund = np.arange(1, fftsize // 2) * sr / fftsize

    a = (len(input) - winsize) % hopsize
    if a:
        input = np.pad(input, (0, hopsize - a), 'constant', constant_values=0)

    raw_frames = np.stack([input[i:i + winsize] for i in range(0, len(input) - winsize + 1, hopsize)], axis=1)
    frame1 = raw_frames * diffwin[:, None]
    frame2 = raw_frames * window[:, None]

    if padsize:
        frame1 = np.pad(frame1, ((0, padsize), (0, 0)), 'constant', constant_values=0)
        frame2 = np.pad(frame2, ((0, padsize), (0, 0)), 'constant', constant_values=0)

    spec1 = rfft(frame1, 0)
    spec2 = rfft(frame2, 0)

    output = spec2.copy()
    powerspec = spec2[2::2] ** 2 + spec2[3::2] ** 2
    output[2::2] = np.sqrt(powerspec)
    output[3::2] = (spec1[3::2] * spec2[2::2] - spec1[2::2] * spec2[3::2]) / powerspec * fac + fund[:, None]

    return output


def addsyn(input, thresh, pitch, scale, hopsize, sr):
    fftsize = input.shape[0]
    bins = fftsize // 2 - 1
    tablen = 10000

    ratio = tablen / sr
    tab = np.sin(np.linspace(0, 2 * np.pi, tablen))
    output = np.empty(hopsize * input.shape[1])

    ampnext = scale * input[2::2]
    freqnext = pitch * input[3::2]
    ampnext = np.hstack((np.zeros((ampnext.shape[0], 1)), ampnext))
    freqnext = np.hstack((np.zeros((freqnext.shape[0], 1)), freqnext))
    x = np.arange(ampnext.shape[1]) * hopsize
    interpa = interpolate.interp1d(x, ampnext, axis=1)
    interpf = interpolate.interp1d(x, freqnext, axis=1)

    lastphase = np.zeros(bins)
    for i in range(input.shape[1]):
        pos = i * hopsize
        x = np.arange(pos, pos + hopsize)
        amp = interpa(x)
        freq = interpf(x) * ratio

        freq[:, 0] += lastphase
        phase = np.cumsum(freq, axis=1).astype(int) % tablen
        lastphase = phase[:, -1]

        outsum = np.take(tab, phase) * amp
        idx = np.where(ampnext[:, i + 1] < thresh)
        outsum[idx] = 0
        output[pos:pos + hopsize] = np.sum(outsum, axis=0)

    return output

if __name__ == '__main__':
    x = np.random.rand(10)
    print(x)
    pv = ifd(x, 4, 1, 10, 'hanning')
    output = addsyn(pv, 0, 1, 1, 4, 1, 10)
    print(output)