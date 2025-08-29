import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass
from tqdm import tqdm
from PIL import Image
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch

# Create a custom binary colormap
n_bins = 100  # Number of bins in the colormap
# Define the color at each bin, with a sharp transition from white to black
values = np.ones((n_bins, 4))  # All white initially
# Set the last 30% of the bins to black
values[-50:, 0:3] = 0  # RGB = black
# Create the colormap
custom_cmap = LinearSegmentedColormap.from_list('binary_white_to_black', values)

def save_image(image, filename, minval=-1.0, maxval=1.0):
    normalized_image = normalize(image, _domain=(minval, maxval), _range=(0.0, 255.0))
    normalized_image = normalized_image.astype(np.uint8)
    Image.fromarray(normalized_image).save(filename)

def get_max_distance_image(image, minval=-1.0, maxval=1.0):
    meanval = (minval + maxval) * 0.5
    max_dist_im = np.full(image.shape, minval, dtype=image.dtype)
    max_dist_im[image < meanval] = maxval
    return max_dist_im

def get_blurred_image(image, sigma=10):
    if len(image.shape) == 4:
        blurred_images = [gaussian_filter(im, (sigma, sigma, 0)) for im in image]
        return np.stack(blurred_images, axis=0)
    elif len(image.shape) == 3:
        return gaussian_filter(image, (sigma, sigma, 0))
    else:
        return gaussian_filter(image, sigma)
    
def get_uniform_image(image, minval=-1.0, maxval=1.0):
    return np.random.uniform(low=minval, high=maxval, size=image.shape)

def get_gaussian_image(image, sigma, mean_image=None, minval=-1.0, maxval=1.0):
    if mean_image is None:
        mean_image = image
    gaussian_image = np.random.randn(*image.shape) * sigma + mean_image
    return np.clip(gaussian_image, a_min=minval, a_max=maxval)

def _get_top_indices(saliency):
    '''
    Returns the indices of saliency ranked in descending order.
    
    Args:
        saliency: A 2D array
    
    Returns:
        An array of shape [np.prod(saliency.shape), 2] representing
        the indices of saliency in descending order of magnitude.
    '''
    return np.flip(np.squeeze(np.dstack(np.unravel_index(np.argsort(saliency.ravel()), 
                                                         saliency.shape))), axis=0)

def ablate_top_k(image, saliency, k, method='mean'):
    '''
    Ablates the top k% pixels in the image as ranked by saliency. 
    
    Args:
        image:    A (width, height, channels) array
        saliency: A (width, height) array of absolute values
        k:        A floating point number between 0.0 and 1.0. The fraction
                  of top pixels to ablate. If the method is `mass_center` or
                  `blur_center`, k instead represents the fraction of the total image to cover.
        method:   One of `mean`, `blur`, `mean_center`, `blur_center`.
        
    Returns:
        An ablated image. Used for interpretability experiments.
    '''
    if k == 0.0:
        return image
    
    ablated_image = image.copy()
    if method == 'mean' or method == 'mean_center':
        baseline_image = np.ones(image.shape) * np.mean(image, axis=(0, 1), keepdims=True)
    elif method == 'blur' or method == 'blur_center':
        baseline_image = get_blurred_image(image, sigma=20.0)
    
    if method == 'mean' or method == 'blur':
        indices     = _get_top_indices(saliency)
        max_to_flip = int(k * indices.shape[0])
        ablated_image[indices[:max_to_flip, 0], 
                      indices[:max_to_flip, 1]] = baseline_image[indices[:max_to_flip, 0],
                                                                 indices[:max_to_flip, 1]]
    elif method == 'mean_center' or method == 'blur_center':
        center_indices = np.array(center_of_mass(np.abs(saliency)))
        lower_bounds   = (center_indices * (1.0 - k)).astype(int)
        upper_bounds   = ((np.array(saliency.shape) - center_indices) * k + center_indices).astype(int)
        
        ablated_image[lower_bounds[0]:upper_bounds[0], 
                      lower_bounds[1]:upper_bounds[1]] = baseline_image[lower_bounds[0]:upper_bounds[0],
                                                                        lower_bounds[1]:upper_bounds[1]]
    return ablated_image
    
def normalize(im_batch, _range=None, _domain=None):
    if len(im_batch.shape) == 2:
        axis = (0, 1)
    elif len(im_batch.shape) == 3:
        axis = (0, 1, 2)
    elif len(im_batch.shape) == 4:
        axis = (1, 2, 3)
    else:
        raise ValueError('im_batch must be of rank 2, 3 or 4')
    
    if _domain is not None:
        min_vals = _domain[0]
        max_vals = _domain[1]
    else:
        min_vals = np.amin(im_batch, axis=axis, keepdims=True)
        max_vals = np.amax(im_batch, axis=axis, keepdims=True)
    
    norm_batch = (im_batch - min_vals) / (max_vals - min_vals)
    
    if _range is not None:
        amin = _range[0]
        amax = _range[1]
        norm_batch = norm_batch * (amax - amin) + amin
    return norm_batch

def norm_clip(x):
    normalized = normalize(x.squeeze())
    clipped = np.clip(normalized, a_min=np.min(normalized), a_max=np.percentile(normalized, 99.9))
    return clipped


# --- Occlusion tutorial ---

# Sliding window occlusion utilities
def sliding_window_positions(H: int, W: int, win: int, stride: int):
    ys = list(range(0, max(1, H - win + 1), stride))
    xs = list(range(0, max(1, W - win + 1), stride))
    if ys[-1] != H - win:
        ys.append(H - win)
    if xs[-1] != W - win:
        xs.append(W - win)
    for y in ys:
        for x in xs:
            yield y, x

def apply_mask(img_hwc: np.ndarray, y: int, x: int, win: int, baseline: float = 0.0):
    out = img_hwc.copy()
    out[y:y+win, x:x+win, :] = baseline
    return out

# Occlusion scan function
# This function performs an occlusion scan on the input image using the specified model.
# It slides a window across the image, applies a mask, and computes the model's predictions
# for each masked image. It returns the results including the position of the window,
# the predicted probability for the target class, and the absolute difference from the true probability.
def occlusion_scan(model, img_hwc: np.ndarray, win: int = 50, stride: int = 50,
                   baseline: float = 0.0, target_class: int = None, device=None):
    if device is None:
        device = next(model.parameters()).device

    x0 = to_chw_tensor(img_hwc).to(device)
    with torch.no_grad():
        p0 = torch.softmax(model(x0), dim=-1)
        if target_class is None:
            target_class = int(p0.argmax(dim=-1))
        p_true = float(p0[0, target_class])

    results = []
    for (y, x) in sliding_window_positions(img_hwc.shape[0], img_hwc.shape[1], win, stride):
        masked = apply_mask(img_hwc, y, x, win, baseline=baseline)
        xt = to_chw_tensor(masked).to(device)
        with torch.no_grad():
            p = torch.softmax(model(xt), dim=-1)[0, target_class].item()
        results.append({'y': y, 'x': x, 'win': win, 'p': p, 'delta': abs(p - p_true), 'image': masked})
    return results, target_class, p_true


# --------CAM tutorial--------
def overlay_cam_on_image(img_tensor_chw: torch.Tensor, cam_hw: np.ndarray, alpha: float = 0.35):
    # img_tensor_chw: [3,224,224], in [0,1]
    img = (img_tensor_chw.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)  # [H,W,3], uint8
    H, W = img.shape[:2]
    # Resize CAM to image size (nearest is fine for simplicity)
    cam_img = Image.fromarray((cam_hw * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0  # [H,W]
    cam_arr = (cam_arr * 255).astype(np.uint8)
    # Create a colored heatmap using simple grayscale-to-RGBA mapping (no external libs)
    # We'll map cam values to the red channel for a clean overlay.
    heatmap = np.zeros_like(img)
    heatmap[..., 0] = cam_arr  # put CAM into red channel
    
    # Blend: img * (1 - alpha) + heatmap * alpha
    blended = (img.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

# Simple de-normalization for visualization
def denormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t * std + mean).clamp(0,1)
# ----------------------------------