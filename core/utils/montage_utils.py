import numpy as np
try:
    from skimage.transform import resize, rescale
except:
    print("Warning: skimage.transform is not available. Will use scipy.misc.imresize instead.")
    from PIL import Image
    def resize(img, size):
        return np.array(Image.fromarray(img).resize(size, Image.Resampling(2)))
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def build_montages(image_list, image_shape, montage_shape, transpose=True):
    """Adapted from imutils.build_montages   add automatic normalization in it.
    ---------------------------------------------------------------------------------------------
    author: Kyle Hounslow
    ---------------------------------------------------------------------------------------------
    Converts a list of single images into a list of 'montage' images of specified rows and columns.
    A new montage image is started once rows and columns of montage image is filled.
    Empty space of incomplete montage images are filled with black pixels
    ---------------------------------------------------------------------------------------------
    :param image_list: python list of input images
    :param image_shape: tuple, size each image will be resized to for display (width, height)
    :param montage_shape: tuple, shape of image montage (width, height)
    :return: list of montage images in numpy array format
    ---------------------------------------------------------------------------------------------
    example usage:
    # load single image
    img = cv2.imread('lena.jpg')
    # duplicate image 25 times
    num_imgs = 25
    img_list = []
    for i in xrange(num_imgs):
        img_list.append(img)
    # convert image list into a montage of 256x256 images tiled in a 5x5 montage
    montages = make_montages_of_images(img_list, (256, 256), (5, 5))
    # iterate through montages and display
    for montage in montages:
        cv2.imshow('montage image', montage)
        cv2.waitKey(0)
    ----------------------------------------------------------------------------------------------
    """
    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')
    image_montages = []
    rowfirst = transpose
    # start with black canvas to draw images onto
    if rowfirst:
        montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3), dtype=np.float64)
    else:
        montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[0]), image_shape[0] *
                                    montage_shape[1], 3), dtype=np.float64)
    cursor_pos = [0, 0]
    start_new_img = False
    for img in image_list:
        # if type(img).__module__ != np.__name__:
        #     raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = resize(img, image_shape)
        if img.dtype in (np.uint8, np.int) and img.max() > 1.0:  # float 0,1 image
            img = (img / 255.0).astype(np.float64)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        if rowfirst:
            cursor_pos[0] += image_shape[0]  # increment cursor x position
            if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
                cursor_pos[1] += image_shape[1]  # increment cursor y position
                cursor_pos[0] = 0
                if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                    cursor_pos = [0, 0]
                    image_montages.append(montage_image)
                    # reset black canvas
                    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3), dtype=np.float64)
                    start_new_img = True
        else:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            if cursor_pos[1] >= image_shape[1] * montage_shape[0]:
                cursor_pos[0] += image_shape[0]  # increment cursor y position
                cursor_pos[1] = 0
                if cursor_pos[0] >= montage_shape[1] * image_shape[0]:
                    cursor_pos = [0, 0]
                    image_montages.append(montage_image)
                    # reset black canvas
                    montage_image = np.zeros(shape=(image_shape[1] * montage_shape[0], image_shape[0] *
                                                    montage_shape[1], 3), dtype=np.float64)
                    start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    return image_montages


def make_grid_np(img_arr, nrow=8, padding=2, pad_value=0, rowfirst=True):
    """ Inspired from make_grid in torchvision.utils"""
    if type(img_arr) is list:
        try:
            img_tsr = np.stack(tuple(img_arr), axis=3)
            img_arr = img_tsr
        except ValueError:
            raise ValueError("img_arr is a list and its elements do not have the same shape as each other.")
    nmaps = img_arr.shape[3]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(img_arr.shape[0] + padding), int(img_arr.shape[1] + padding)
    grid = np.zeros((height * ymaps + padding, width * xmaps + padding, 3), dtype=img_arr.dtype)
    grid.fill(pad_value)
    k = 0
    if rowfirst:
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                grid[y * height + padding: (y + 1) * height, x * width + padding: (x + 1) * width, :] = img_arr[:,:,:,k]
                k = k + 1
    else:
        for x in range(xmaps):
            for y in range(ymaps):
                if k >= nmaps:
                    break
                grid[y * height + padding: (y + 1) * height, x * width + padding: (x + 1) * width, :] = img_arr[:,:,:,k]
                k = k + 1
    return grid


import torch
import math
import warnings
from typing import Union, List, Optional, Tuple
from torchvision.utils import make_grid
def make_grid_T(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    rowfirst: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
        rowfirst (bool, optional): If ``True``, the images fill the grid in row-first order,
                                otherwise in column-first order. Default: ``True``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    if rowfirst:
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
    else:
        ymaps = min(nrow, nmaps)
        xmaps = int(math.ceil(float(nmaps) / ymaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    if rowfirst:
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                # Tensor.copy_() is a valid method but seems to be missing from the stubs
                # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
                grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                    2, x * width + padding, width - padding
                ).copy_(tensor[k])
                k = k + 1
    else:
        for x in range(xmaps):
            for y in range(ymaps):
                if k >= nmaps:
                    break
                grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                    2, x * width + padding, width - padding
                ).copy_(tensor[k])
                k = k + 1
    return grid

def color_frame(img, color, pad=10):
    outimg = np.ones((img.shape[0] + pad * 2, img.shape[1] + pad * 2, 3))
    outimg = outimg * color[:3]
    outimg[pad:-pad, pad:-pad, :] = img
    return outimg


def color_framed_montages(image_list, image_shape, montage_shape, scores, cmap=plt.cm.summer, pad=24, vmin=None, vmax=None):
    # get color for each cell
    if (not scores is None) and (not cmap is None):
        lb = np.min(scores) if vmin is None else vmin
        ub = max(np.max(scores), lb + 0.001) if vmax is None else vmax
        colorlist = [cmap((score - lb) / (ub - lb)) for score in scores]
        # pad color to the image
        frame_image_list = [color_frame(img, color, pad=pad) for img, color in zip(image_list, colorlist)]
    else:
        frame_image_list = image_list
    image_montages = []
    # start with black canvas to draw images onto
    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                             dtype=np.float64)
    cursor_pos = [0, 0]
    start_new_img = False
    for img in frame_image_list:
        if type(img).__module__ != np.__name__:
            raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = resize(img, image_shape)
        if img.dtype in (np.uint8, np.int) and img.max() > 1.0:  # float 0,1 image
            img = (img / 255.0).astype(np.float64)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = np.zeros(
                    shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3), dtype=np.float64)
                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    return image_montages



def crop_from_montage(img, imgid: Union[Tuple, int] = (0,0), imgsize=256, pad=2):
    nrow, ncol = (img.shape[0] - pad) // (imgsize + pad), (img.shape[1] - pad) // (imgsize + pad)
    if imgid == "rand":  imgid = np.random.randint(nrow * ncol)
    elif type(imgid) is tuple:
        ri, ci = imgid
    elif type(imgid) is int:
        if imgid < 0:
            imgid = nrow * ncol + imgid
            ri, ci = np.unravel_index(imgid, (nrow, ncol))
        else:
            ri, ci = np.unravel_index(imgid, (nrow, ncol))
    else:
        raise Exception("imgid must be tuple or int")
    img_crop = img[pad + (pad+imgsize)*ri:pad + imgsize + (pad+imgsize)*ri, \
                   pad + (pad+imgsize)*ci:pad + imgsize + (pad+imgsize)*ci, :]
    return img_crop


def crop_all_from_montage(img, totalnum=None, imgsize=512, pad=2, autostop=True):
    """Return all crops from a montage image"""
    nrow, ncol = (img.shape[0] - pad) // (imgsize + pad), (img.shape[1] - pad) // (imgsize + pad)
    if totalnum is None:
        totalnum = nrow * ncol
    imgcol = []
    for imgid in range(totalnum):
        ri, ci = np.unravel_index(imgid, (nrow, ncol))
        img_crop = img[pad + (pad + imgsize) * ri:pad + imgsize + (pad + imgsize) * ri, \
               pad + (pad + imgsize) * ci:pad + imgsize + (pad + imgsize) * ci, :]
        if np.allclose(img_crop, np.zeros(1)) and autostop:
            break
        imgcol.append(img_crop)
    return imgcol


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
def get_first_non_none_image_size(image_array):
    for img in image_array.flatten():
        if img is not None:
            return img.size
    return (0, 0)  # Return a default size if all images are None


# make these images as a grid. PIL to image grid
def PIL_array_to_montage(image_array):
    """Create a grid of images from a 2D array of PIL Images
    
    Args:
        image_array: 2D numpy array of PIL Image objects
        
    Returns:
        PIL Image containing the grid of images
    """
    # Calculate dimensions for the grid
    n_rows, n_cols = image_array.shape
    if n_rows == 0 or n_cols == 0:
        return None
    
    img_width, img_height = get_first_non_none_image_size(image_array)
    # Create a new blank image with space for all images
    grid_img = Image.new('RGB', (n_cols * img_width, n_rows * img_height))
    # Paste each image into the grid
    for i in range(n_rows):
        for j in range(n_cols):
            if image_array[i,j] is not None:
                grid_img.paste(image_array[i,j], (j * img_width, i * img_height))
            else:
                print(f"Warning: Image at ({i}, {j}) is None")
            
    return grid_img


# make these images as a grid. PIL to image grid
def PIL_array_to_montage_score_frame(image_array, score_array, colormap='viridis', border_size=10, clim=None):
    """Create a grid of images from a 2D array of PIL Images with colored frames based on scores.
    
    Args:
        image_array: 2D numpy array of PIL Image objects.
        score_array: 2D numpy array of scores corresponding to each image.
        colormap: str, name of the matplotlib colormap to use for mapping scores to colors.
        border_size: int, size of the colored frame around each image in pixels.
        
    Returns:
        PIL Image containing the grid of images with colored frames.
    """
    # Ensure score_array is a numpy array
    if not isinstance(score_array, np.ndarray):
        score_array = np.array(score_array)
    n_rows, n_cols = image_array.shape
    if n_rows == 0 or n_cols == 0:
        return None
    # Normalize scores to [0, 1] for colormap mapping
    if clim is None:
        clim = (np.min(score_array), np.max(score_array))
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    mapped_scores = norm(score_array)
    # Get the colormap
    cmap = plt.get_cmap(colormap)
    # Map scores to colors without manual normalization
    colors = [tuple(int(255 * c) for c in cmap(score)[:3]) for score in mapped_scores.flatten()]
    colors = np.array(colors).reshape(score_array.shape + (3,))
    # Calculate dimensions for the grid with borders
    img_width, img_height = get_first_non_none_image_size(image_array)
    bordered_width = img_width + 2 * border_size
    bordered_height = img_height + 2 * border_size
    # Create a new blank image with space for all images and their borders
    grid_img = Image.new('RGB', (n_cols * bordered_width, n_rows * bordered_height))
    # Paste each image with colored border into the grid
    for i in range(n_rows):
        for j in range(n_cols):
            img = image_array[i, j]
            if img is not None:
                color = tuple(colors[i, j])
                # Create a new image with border
                bordered_img = Image.new('RGB', (bordered_width, bordered_height), color)
                bordered_img.paste(img, (border_size, border_size))
                # Paste the bordered image into the grid
                grid_img.paste(bordered_img, (j * bordered_width, i * bordered_height))
            else:
                print(f"Warning: Image at ({i}, {j}) is None")
    
    return grid_img
