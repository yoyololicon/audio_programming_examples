from ch8.stft import *


def simplp(input):
    output = input.copy()
    output[0] = 1
    output[1] = 0
    fftsize = input.shape[0]
    filt = np.cos(np.arange(1, fftsize // 2 + 1) * np.pi / fftsize)

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

    w = np.arange(1, fftsize) * np.pi * 2 / fftsize
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
    output[1] *= (1 + radius) / (1 + rad2 + radsq)

    w = np.arange(1, fftsize) * np.pi * 2 * delay / fftsize
    sinw = np.sin(w)
    cosw = np.cos(w)

    div = 1 - rad2 * cosw + radsq
    re = (cosw - radius) / div
    im = (sinw - rad2 * cosw * sinw) / div

    output[2::2] = input[2::2] * re - input[3::2] * im
    output[3::2] = input[3::2] * re + input[2::2] * im

    return output * scale
