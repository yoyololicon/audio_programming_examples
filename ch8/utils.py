import numpy as np
from scipy import signal, fftpack


def check_pow_2(size):
    return sum([int(x) for x in bin(size)[2:]]) == 1


def rfft(x, axis=-1):
    if not check_pow_2(x.shape[axis]):
        print("Input size not power of 2.")

    y = fftpack.rfft(x, axis=axis)
    y = np.roll(y, 1, axis=axis)

    slc = [slice(None)] * len(y.shape)
    slc[axis] = slice(0, 2)
    slc2 = slc.copy()
    slc2[axis] = slice(1, None, -1)

    y[tuple(slc)] = y[tuple(slc2)]

    return y


def irfft(x, axis=-1):
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(0, 2)
    slc2 = slc.copy()
    slc2[axis] = slice(1, None, -1)

    x[tuple(slc)] = x[tuple(slc2)]

    x = np.roll(x, -1, axis=axis)
    y = fftpack.irfft(x, axis=axis)

    return y


def stft(input, fftsize, hopsize, window=None):
    # assume fftsize always a power of 2
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

    a = (len(input) - winsize) % hopsize
    if a:
        input = np.pad(input, (0, hopsize - a), 'constant', constant_values=0)

    raw_frames = np.stack([input[i:i + winsize] for i in range(0, len(input) - winsize + 1, hopsize)], axis=1)
    raw_frames *= window[:, None]

    if padsize:
        raw_frames = np.pad(raw_frames, ((0, padsize), (0, 0)), 'constant', constant_values=0)

    spec = rfft(raw_frames, 0)
    return spec


def istft(input, hopsize, windowtype=None):
    fftsize, t = input.shape
    if type(windowtype) == str:
        window = signal.get_window(windowtype, fftsize)
    else:
        window = np.ones(fftsize)

    frames = irfft(input, 0)
    frames *= window[:, None]

    output = np.zeros(hopsize * t + fftsize - hopsize)
    for i in range(t):
        pos = i * hopsize
        output[pos:pos + fftsize] += frames[:, i]
    return output


def simplp(input):
    output = input.copy()
    output[0] = 1
    output[1] = 0
    fftsize = input.shape[0]
    filt = np.cos(np.arange(1, fftsize // 2) * np.pi / fftsize)

    mag = np.sqrt(input[2::2] ** 2 + input[3::2] ** 2)
    phi = np.arctan2(input[3::2], input[2::2])
    mag *= filt[:, None]
    output[2::2] = mag * np.cos(phi)
    output[3::2] = mag * np.sin(phi)
    return output


def simphp(input):
    output = input.copy()
    output[0] = 1
    output[1] = 0
    fftsize = input.shape[0]
    filt = np.sin(np.arange(1, fftsize // 2) * np.pi / fftsize)

    mag = np.sqrt(input[2::2] ** 2 + input[3::2] ** 2)
    phi = np.arctan2(input[3::2], input[2::2])
    mag *= filt[:, None]
    output[2::2] = mag * np.cos(phi)
    output[3::2] = mag * np.sin(phi)
    return output


def specreson(input, scale, angle, radius):
    fftsize = input.shape[0]
    costheta = np.cos(angle)
    radsq = radius ** 2
    rad2 = radius * 2

    output = input.copy()
    output[0] /= 1 - rad2 * costheta + radsq
    output[1] /= 1 + rad2 * costheta + radsq

    w = np.arange(1, fftsize // 2) * np.pi * 2 / fftsize
    sinw = np.sin(w)
    cosw = np.cos(w)
    cos2w = np.cos(2 * w)

    re = 1 - rad2 * costheta * cosw + radsq * cos2w
    im = sinw * (rad2 * costheta - 2 * radsq * cosw)

    div = re ** 2 + im ** 2
    output[2::2] = (input[2::2] * re + input[3::2] * im) / div
    output[3::2] = (input[3::2] * re - input[2::2] * im) / div

    output *= scale
    return output


def specomb(input, scale, delay, radius, sr):
    fftsize = input.shape[0]
    radsq = radius ** 2
    rad2 = radius * 2
    delay *= sr

    output = input.copy()
    output[0] *= (1 - radius) / (1 - rad2 + radsq)
    output[1] *= -(1 + radius) / (1 + rad2 + radsq)

    w = np.arange(1, fftsize // 2) * np.pi * 2 * delay / fftsize
    sinw = np.sin(w)
    cosw = np.cos(w)

    div = 1 - rad2 * cosw + radsq
    re = (cosw - radius) / div
    im = (sinw - rad2 * cosw * sinw) / div

    output[2::2] = input[2::2] * re - input[3::2] * im
    output[3::2] = input[3::2] * re + input[2::2] * im

    return output * scale


if __name__ == '__main__':
    import librosa

    y, sr = librosa.load('male.wav', sr=None)

    spec = stft(y, 512, 128, 'hanning')
    print(spec.shape)

    librosa.output.write_wav('test.wav', istft(spec, 128, 'hanning'), sr)
