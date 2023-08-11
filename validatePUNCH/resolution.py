from typing import Tuple, Optional

import numpy as np
import scipy.fft
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def radial_profile(data: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
    """Compute a radial profile of some data centered at a specific point

    Parameters
    ----------
    data : np.ndarray
        image to compute the radial profile of
    center : Tuple[int, int]
        where the radial profile is centered in the image

    Returns
    -------
    np.ndarray
        radial profile of an image
    """
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    profile = tbin / nr
    return profile


def measure_fwhm_by_fft(image_cutout: np.ndarray,
                        show_fig: bool = False,
                        fig_save_path: Optional[str] = None,
                        vmin_percent: float = 3,
                        vmax_percent: float = 99,
                        with_interp: bool = True) -> float:
    """Measure the FWHM resolution of an image cutout, assumed to be source centered

    Parameters
    ----------
    image_cutout: np.ndarray
        the image to measure the profile in. assumed the star-like source is at the image center
    show_fig : bool
        whether to show a figure corresponding to the fit
    fig_save_path : Optional[str]
        where to save the shown figure, requires that `show_fig=True`
    vmin_percent : float
        lower percentile of the image_cutout to show in the plot
    vmax_percent : float
        upper percentile of the image_cutout to show in the plot
    with_interp : bool
        whether to interpolate when measuring the FWHM

    Returns
    -------
    float
        FWHM of the sample
    """
    # compute the mtf
    mtf = np.abs(scipy.fft.fftshift(scipy.fft.fft2(image_cutout)))
    mtf[int(mtf.shape[0] / 2), int(mtf.shape[1] / 2)] = 0
    mtf = mtf / np.max(mtf)

    # figure out the frequenies and resolution x_axis terms
    f = scipy.fft.fftshift(scipy.fft.fftfreq(mtf.shape[0]))
    ff = f[f.shape[0] // 2:]
    x_axis = 1 / ff / (2 * np.pi) * 1.5 * 2

    # compute the radial profile
    rp = radial_profile(mtf, (mtf.shape[0] / 2, mtf.shape[1] / 2))[:len(ff)]
    rp = rp / np.max(rp)

    # drop the zero component
    rp = rp[1:]
    ff = ff[1:]
    x_axis = x_axis[1:]

    # find the half max
    mi = np.argmin(np.abs(rp - 0.5))
    resolution = x_axis[mi]

    if with_interp:
        cs = CubicSpline(ff, rp)
        new_ff = np.linspace(ff[0], ff[-1], 100)
        new_rp = cs(new_ff)
        new_x_axis = 1 / new_ff / (2 * np.pi) * 1.5 * 2
        new_mi = np.argmin(np.abs(new_rp - 0.5))
        resolution = new_x_axis[new_mi]

    # everything remaining is figure generation code, could be moved into separate function
    if show_fig:
        fig = plt.figure(constrained_layout=True, figsize=(6, 6))
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, :])

        im = ax0.imshow(image_cutout,
                        vmin=np.nanpercentile(image_cutout, vmin_percent),
                        vmax=np.nanpercentile(image_cutout, vmax_percent),
                        cmap='Greys_r',
                        origin='lower')
        ax0.set_title("Source patch")
        cbaxes = inset_axes(ax0, loc=4, width="10%", height="70%")
        cbar = fig.colorbar(im, ax=cbaxes, shrink=0.6)
        cbar.ax.tick_params(labelsize=8, labelcolor='red')

        im = ax1.imshow(mtf, vmin=0, vmax=0.9, origin='lower')
        ax1.set_title("FFT of source patch")
        fig.colorbar(im, ax=ax1, shrink=0.6, )

        ax2.plot(x_axis, rp, 'b.-')
        if with_interp:
            ax.plot(new_x_axis, new_rp, 'y-')
            ax2.plot(new_x_axis[new_mi], new_rp[new_mi], 'ro')
        else:
            ax2.plot(x_axis[mi], rp[mi], 'ro')
        ax2.set_title(f"fwhm: {resolution: 0.2f} arcmin")
        ax2.set_xlabel("Resolution [arcmin]")
        ax2.set_ylabel("Norm. response")
        ax2.set_ylim(0, 1.05)

        ax3.plot(ff, rp, 'b.-')
        if with_interp:
            ax3.plot(new_ff, new_rp, 'y-')
            ax3.plot(new_ff[new_mi], new_rp[new_mi], 'ro')
        else:
            ax3.plot(ff[mi], rp[mi], 'ro')
        ax3.set_xlabel("Frequency")
        ax3.set_ylabel("Normal. response")
        ax3.set_ylim(0, 1.05)

        fig.show()
        if fig_save_path:
            fig.savefig(fig_save_path)
            plt.close()

    return resolution
