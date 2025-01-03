from scipy import stats
import numpy as np
from scipy import ndimage
import torch

def power_spectrum_freq(grey_image):
    
    image = grey_image
    # image[:,:] *=0
    # image[::2,::2] +=1
    npix = image.shape[0]

    print(image.min(), image.max())

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)


    return kvals, Abins


def power_spectrum_orientation(grey_image):

    # Perform a 2D Fourier Transform
    fourier_image = np.fft.fft2(grey_image)
    fourier_amplitudes = np.abs(fourier_image) ** 2

    # Calculate the orientation spectrum
    orientations_degrees = np.linspace(0, 180, num=180, endpoint=False)  # Degrees
    orientations_radians = np.radians(orientations_degrees)  # Convert to radians
    orientation_spectrum = np.zeros(orientations_radians.shape)

    for i, theta in enumerate(orientations_radians):
        # Apply Radon transform for each orientation
        rotated_image = ndimage.rotate(grey_image, np.degrees(theta), reshape=False)
        projection = np.sum(rotated_image, axis=1)
        orientation_spectrum[i] = np.sum(projection ** 2)

    # Normalize the orientation spectrum
    orientation_spectrum /= np.max(orientation_spectrum)
    # print(orientation_spectrum.sum())
    return orientations_degrees, orientation_spectrum




# Calculate the RMS contrast for each color channel
def calculate_single_contrast(channel):
    mean_pixel_value = torch.mean(channel)
    return torch.sqrt(torch.mean((channel - mean_pixel_value) ** 2))



def calculate_rms_contrast(rgb_image):

    r_channel , g_channel, b_channel = rgb_image[0], rgb_image[1], rgb_image[2]

    rms_contrast_r = calculate_single_contrast(r_channel)
    rms_contrast_g = calculate_single_contrast(g_channel)
    rms_contrast_b = calculate_single_contrast(b_channel)

    # Convert the RMS contrast to NumPy arrays for easy printing
    rms_contrast_r = rms_contrast_r.item()
    rms_contrast_g = rms_contrast_g.item()
    rms_contrast_b = rms_contrast_b.item()

    # Display the RMS contrast values for each color channel
    # print(f"RMS Contrast (Red): {rms_contrast_r}")
    # print(f"RMS Contrast (Green): {rms_contrast_g}")
    # print(f"RMS Contrast (Blue): {rms_contrast_b}")

    # print(f"MEAN RMS Contrast : {(rms_contrast_b + rms_contrast_g + rms_contrast_r)/3.}")

    return (rms_contrast_b + rms_contrast_g + rms_contrast_r)/3.