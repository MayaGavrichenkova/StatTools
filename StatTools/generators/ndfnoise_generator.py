"""Generate n-dimensional field."""

import numpy as np


def ndfnoise(shape: tuple[int], hurst: float, normalize: bool = True) -> np.ndarray:
    """
    Generate n-dimensional fractional Gaussian noise using the spectral synthesis method.
    #TODO: fBm or fGm?

    Args:
        shape (tuple[int]): Shape of the field, example: (256, 256) or (64, 64, 64)
        hurst (float): Hurst exponent H, 0.5 < H < 1.5
        normalize (bool): If True, normalize the field to have zero mean and unit variance.
                          Default is True.

    Returns:
        field (np.ndarray): n-dimensional fractional Gaussian noise field.

    Basic usage:
        ```python
        f = ndfnoise((2048, 2048), hurst=0.8)
        ```
    """
    shape = tuple(shape)
    dim = len(shape)

    # Частотные сетки (k1, k2, ..., kd)
    freq_grids = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")

    k_sq = np.zeros(shape)
    for fr in freq_grids:
        k_sq += fr**2
    k = np.sqrt(k_sq)

    # Энергетический спектр S(k) ∝ |k|^{-(H + d/2)}
    alpha = hurst + dim / 2.0
    S = np.zeros_like(k)
    nonzero = k > 0
    S[nonzero] = k[nonzero] ** (-alpha)

    # Случайные фазы + амплитуды
    noise_real = np.random.normal(size=shape)
    noise_imag = np.random.normal(size=shape)
    noise_complex = noise_real + 1j * noise_imag

    # Применяем спектр
    F = noise_complex * S

    # Обратное FFT
    field = np.fft.ifftn(F).real

    # Normalization
    if normalize:
        field -= field.mean()
        field /= field.std() + 1e-12

    return field


def ndfnoise_optimized(
    shape: tuple[int], hurst: float, normalize: bool = True, dtype=np.float32
) -> np.ndarray:
    """
    Optimized n-dimensional fractional Gaussian noise (fGn) generator.
    Uses rFFT (real FFT) and float32 to reduce memory usage.

    Args:
        shape (tuple[int]): Shape of the field, example: (256, 256) or (64, 64, 64)
        hurst (float): Hurst exponent H, 0.5 < H < 1.5
        normalize (bool): If True, normalize the field to have zero mean and unit variance.
                          Default is True.
        dtype (np.dtype): Data type of the field. Default is float32.

    Returns:
        field (np.ndarray): n-dimensional fractional Gaussian noise field.

    Basic usage:
        ```python
        f = ndfnoise((2048, 2048), hurst=0.8)
        ```
    """

    shape = tuple(shape)
    dim = len(shape)

    freq_axes = [np.fft.fftfreq(n, d=1.0).astype(dtype) for n in shape[:-1]]
    freq_last = np.fft.rfftfreq(shape[-1], d=1.0).astype(dtype)

    grids = np.meshgrid(*freq_axes, freq_last, indexing="ij")

    k_sq = np.zeros([*shape[:-1], shape[-1] // 2 + 1], dtype=dtype)
    for g in grids:
        k_sq += g * g
    k = np.sqrt(k_sq)

    alpha = hurst + dim / 2
    S = np.zeros_like(k, dtype=dtype)
    nonzero = k > 0
    S[nonzero] = k[nonzero] ** (-alpha)

    noise_real = np.random.normal(0.0, 1.0, S.shape).astype(dtype)
    noise_imag = np.random.normal(0.0, 1.0, S.shape).astype(dtype)
    noise_complex = noise_real + 1j * noise_imag

    F = noise_complex * S

    field = np.fft.irfftn(F, s=shape).astype(dtype)

    # Normalize
    if normalize:
        field -= field.mean()
        std = field.std()
        if std > 0:
            field /= std

    return field
