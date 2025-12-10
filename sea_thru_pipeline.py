# sea_thru_pipeline.py
import os
import sys
import math
import time
import warnings
import collections
import numpy as np
from PIL import Image
import scipy as sp
import scipy.optimize
import scipy.stats
from skimage.restoration import denoise_bilateral
from skimage.morphology import closing, square, disk
# compatibility wrappers for skimage API changes
from skimage.restoration import denoise_tv_chambolle as _denoise_tv_chambolle
from skimage.restoration import estimate_sigma as _estimate_sigma

# ----------------------------
# Compatibility helpers
# ----------------------------
def _estimate_sigma_compat(img, **kwargs):
    kwargs_copy = dict(kwargs)
    if 'multichannel' in kwargs_copy:
        try:
            return _estimate_sigma(img, **kwargs_copy)
        except TypeError:
            kwargs_copy.pop('multichannel', None)
    try:
        return _estimate_sigma(img, channel_axis=-1, **{k: v for k, v in kwargs_copy.items() if k != 'multichannel'})
    except TypeError:
        return _estimate_sigma(img, multichannel=True, **{k: v for k, v in kwargs_copy.items() if k != 'channel_axis'})

def _denoise_tv_chambolle_compat(img, weight, **kwargs):
    kwargs_copy = dict(kwargs)
    if 'multichannel' in kwargs_copy:
        try:
            return _denoise_tv_chambolle(img, weight, **kwargs_copy)
        except TypeError:
            kwargs_copy.pop('multichannel', None)
    try:
        return _denoise_tv_chambolle(img, weight, channel_axis=-1, **{k: v for k, v in kwargs_copy.items() if k != 'multichannel'})
    except TypeError:
        return _denoise_tv_chambolle(img, weight, multichannel=True, **{k: v for k, v in kwargs_copy.items() if k != 'channel_axis'})

# ----------------------------
# Small numeric helpers
# ----------------------------
_eps = 1e-8

def scale01(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < _eps:
        return np.clip(arr - mn, 0.0, 1.0)
    return (arr - mn) / (mx - mn)

# ----------------------------
# Sea-Thru building blocks (vectorized where practical)
# ----------------------------

def find_backscatter_estimation_points(img, depths, num_bins=10, fraction=0.01, max_vals=20, min_depth_percent=0.0):
    """
    img: HxWx3 float [0..1]
    depths: HxW float (meters or relative)
    returns arrays of shape (N,2): (depth, val)
    """
    z_max, z_min = np.nanmax(depths), np.nanmin(depths)
    min_depth = z_min + (min_depth_percent * (z_max - z_min))
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.mean(img, axis=2)
    points_r = []
    points_g = []
    points_b = []
    h,w = depths.shape
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i+1]
        mask = (depths > min_depth) & (depths >= a) & (depths <= b)
        if not np.any(mask):
            continue
        norms_in_range = img_norms[mask]
        px_in_range = img[mask]
        depths_in_range = depths[mask]
        idxs = np.argsort(norms_in_range)
        n_take = min(math.ceil(fraction * len(idxs)), max_vals)
        sel = idxs[:n_take]
        if sel.size == 0:
            continue
        sel_depths = depths_in_range[sel]
        sel_px = px_in_range[sel]
        points_r.extend([(z, p[0]) for z,p in zip(sel_depths, sel_px)])
        points_g.extend([(z, p[1]) for z,p in zip(sel_depths, sel_px)])
        points_b.extend([(z, p[2]) for z,p in zip(sel_depths, sel_px)])
    def to_arr(lst):
        if len(lst) == 0:
            return np.zeros((0,2), dtype=np.float32)
        return np.array(lst, dtype=np.float32)
    return to_arr(points_r), to_arr(points_g), to_arr(points_b)

def find_backscatter_values(B_pts, depths, restarts=10, max_mean_loss_fraction=0.1):
    """
    Fit model:
    B(z) = B_inf * (1 - exp(-beta_B * z)) + J' * exp(-beta_D' * z)
    If not converging, fallback to linear fit.
    """
    if B_pts.shape[0] == 0:
        # fallback: zeros
        return np.zeros_like(depths), np.array([0.,0.])
    B_vals = B_pts[:,1]
    B_depths = B_pts[:,0]
    z_max, z_min = np.nanmax(depths), np.nanmin(depths)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    best_loss = np.inf
    coefs = None
    def estimate(z, B_inf, beta_B, J_prime, beta_D_prime):
        return (B_inf * (1 - np.exp(-beta_B * z))) + (J_prime * np.exp(-beta_D_prime * z))
    def loss(params):
        return np.mean(np.abs(B_vals - estimate(B_depths, *params)))
    bounds_lower = [0,0,0,0]
    bounds_upper = [1,5,1,5]
    for _ in range(restarts):
        try:
            p0 = np.random.random(4) * bounds_upper
            optp, _ = sp.optimize.curve_fit(estimate, B_depths, B_vals, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=2000)
            L = loss(optp)
            if L < best_loss:
                best_loss = L
                coefs = optp
        except Exception as e:
            # ignore failing restarts
            pass
    if coefs is None or best_loss > max_mean_loss:
        # linear fallback
        try:
            slope, intercept, *_ = sp.stats.linregress(B_depths, B_vals)
            BD = (slope * depths) + intercept
            return BD, np.array([slope, intercept])
        except Exception:
            return np.zeros_like(depths), np.array([0.,0.])
    return estimate(depths, *coefs), coefs

def construct_neighborhood_map(depths, epsilon=0.05):
    """
    Flood-fill based neighborhood grouping based on depth similarity.
    Returns nmap (H,W) with labels (0 means background) and num_labels
    """
    eps = (np.nanmax(depths) - np.nanmin(depths)) * epsilon
    h,w = depths.shape
    nmap = np.zeros_like(depths, dtype=np.int32)
    label = 1
    # indices of unassigned
    unassigned = np.where(nmap==0)
    # We'll do iterative flood fills: use stack to avoid recursion
    for i in range(h):
        for j in range(w):
            if nmap[i,j] != 0:
                continue
            # start flood
            stack = [(i,j)]
            while stack:
                x,y = stack.pop()
                if not (0<=x<h and 0<=y<w):
                    continue
                if nmap[x,y] != 0:
                    continue
                if abs(depths[x,y] - depths[i,j]) <= eps:
                    nmap[x,y] = label
                    # neighbors
                    if x+1 < h: stack.append((x+1,y))
                    if x-1 >= 0: stack.append((x-1,y))
                    if y+1 < w: stack.append((x,y+1))
                    if y-1 >= 0: stack.append((x,y-1))
            label += 1
    # reset largest background region to 0 (as in original)
    uniq, counts = np.unique(nmap[depths==0], return_counts=True)
    if uniq.size > 0:
        largest_label = uniq[np.argmax(counts)]
        nmap[nmap==largest_label] = 0
    return nmap, label-1

def find_closest_label(nmap, sx, sy):
    h,w = nmap.shape
    mask = np.zeros_like(nmap, dtype=bool)
    q = collections.deque()
    q.append((sx,sy))
    while q:
        x,y = q.popleft()
        if not (0<=x<h and 0<=y<w):
            continue
        if nmap[x,y] != 0:
            return nmap[x,y]
        mask[x,y] = True
        if x+1 < h and not mask[x+1,y]: q.append((x+1,y))
        if x-1 >=0 and not mask[x-1,y]: q.append((x-1,y))
        if y+1 < w and not mask[x,y+1]: q.append((x,y+1))
        if y-1 >=0 and not mask[x,y-1]: q.append((x,y-1))
    return 0

def refine_neighborhood_map(nmap, min_size=10, radius=3):
    refined = np.zeros_like(nmap)
    vals, counts = np.unique(nmap, return_counts=True)
    # sort by size desc
    order = np.argsort(-counts)
    label_id = 1
    for idx in order:
        val = vals[idx]
        cnt = counts[idx]
        if val == 0:
            continue
        if cnt >= min_size:
            refined[nmap==val] = label_id
            label_id += 1
    # assign small regions to nearest big region
    for idx in order:
        val = vals[idx]
        cnt = counts[idx]
        if val == 0 or cnt >= min_size:
            continue
        coords = np.column_stack(np.where(nmap==val))
        for (x,y) in coords:
            refined[x,y] = find_closest_label(refined, x, y)
    # morphological closing to remove holes
    refined = closing(refined, square(radius))
    return refined, label_id-1

def estimate_illumination(img_channel, B_channel, neighborhood_map, num_neighborhoods, p=0.5, f=2.0, max_iters=100, tol=1e-5):
    """
    Estimate local-space averaged illuminant map per channel.
    img_channel and B_channel are HxW floats.
    neighborhood_map labeled regions from 1..N
    """
    D = img_channel - B_channel
    avg_cs = np.zeros_like(img_channel, dtype=np.float32)
    avg_cs_prime = np.zeros_like(avg_cs)
    locs_list = [None] * num_neighborhoods
    sizes = np.zeros(num_neighborhoods, dtype=np.int32)
    for label in range(1, num_neighborhoods+1):
        locs = np.where(neighborhood_map == label)
        locs_list[label-1] = locs
        sizes[label-1] = np.size(locs[0])
    for _ in range(max_iters):
        for label in range(1, num_neighborhoods+1):
            locs = locs_list[label-1]
            size = sizes[label-1] - 1
            if size <= 0:
                continue
            # compute avg_cs_prime at locations
            s = np.sum(avg_cs[locs]) - avg_cs[locs]
            avg_cs_prime[locs] = s / max(size,1)
        new_avg_cs = (D * p) + (avg_cs_prime * (1 - p))
        if np.max(np.abs(avg_cs - new_avg_cs)) < tol:
            avg_cs = new_avg_cs
            break
        avg_cs = new_avg_cs
    # bilateral denoise and scale by f
    illum = f * denoise_bilateral(np.maximum(0, avg_cs))
    return illum

def estimate_wideband_attentuation(depths, illum, radius=6, max_val=10.0):
    eps = 1E-8
    BD = np.minimum(max_val, -np.log(illum + eps) / (np.maximum(0, depths) + eps))
    mask = (depths > eps) & (illum > eps)
    refined = denoise_bilateral(closing(np.maximum(0, BD * mask), disk(radius)))
    return refined, []

def calculate_beta_D(depths, a, b, c, d):
    return (a * np.exp(b * depths)) + (c * np.exp(d * depths))

def filter_data(X, Y, radius_fraction=0.01):
    if X.size == 0:
        return np.array([]), np.array([])
    idxs = np.argsort(X)
    Xs = X[idxs]
    Ys = Y[idxs]
    x_max, x_min = np.max(Xs), np.min(Xs)
    radius = (radius_fraction * (x_max - x_min))
    ds = np.cumsum(Xs - np.roll(Xs, 1))
    dX = [Xs[0]]
    dY = [Ys[0]]
    tempX = []
    tempY = []
    pos = 0
    for i in range(1, ds.shape[0]):
        if ds[i] - ds[pos] >= radius:
            tempX.append(Xs[i])
            tempY.append(Ys[i])
            idxs2 = np.argsort(tempY)
            med_idx = len(idxs2) // 2
            dX.append(tempX[med_idx])
            dY.append(tempY[med_idx])
            pos = i
        else:
            tempX.append(Xs[i])
            tempY.append(Ys[i])
    return np.array(dX), np.array(dY)

def refine_wideband_attentuation(depths, illum, estimation, restarts=10, min_depth_fraction=0.1, max_mean_loss_fraction=np.inf, l=1.0, radius_fraction=0.01):
    eps = 1E-8
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_fraction * (z_max - z_min))
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf
    locs_mask = (illum > 0) & (depths > min_depth) & (estimation > eps)
    if not np.any(locs_mask):
        return estimation, np.array([0,0])
    def calc_reconst(depths_loc, illum_loc, a, b, c, d):
        return -np.log(illum_loc + eps) / (calculate_beta_D(depths_loc, a, b, c, d) + eps)
    def loss(a,b,c,d):
        return np.mean(np.abs(depths[locs_mask] - calc_reconst(depths[locs_mask], illum[locs_mask], a,b,c,d)))
    dX, dY = filter_data(depths[locs_mask], estimation[locs_mask], radius_fraction)
    if dX.size == 0:
        return estimation, np.array([0,0])
    for _ in range(restarts):
        try:
            p0 = np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.])
            optp, _ = sp.optimize.curve_fit(calculate_beta_D, dX, dY, p0=p0, bounds=([0,-100,0,-100],[100,0,100,0]), maxfev=2000)
            L = loss(*optp)
            if L < best_loss:
                best_loss = L
                coefs = optp
        except Exception:
            pass
    if coefs is None or best_loss > max_mean_loss:
        # linear fallback
        try:
            slope, intercept, *_ = sp.stats.linregress(depths[locs_mask], estimation[locs_mask])
            BD = l * (slope * depths + intercept)
            return BD, np.array([slope, intercept])
        except Exception:
            return estimation, np.array([0,0])
    BD = l * calculate_beta_D(depths, *coefs)
    return BD, coefs

def wbalance_no_red_10p(img):
    # img HxWx3 in [0..1]
    flat = img.reshape(-1,3)
    n = flat.shape[0]
    topk = int(round(-0.1 * n))
    if topk == 0:
        topk = 1
    dr = 1.0 / np.mean(np.sort(flat[:,1])[topk:])
    db = 1.0 / np.mean(np.sort(flat[:,2])[topk:])
    dsum = dr + db
    dg = dr / dsum * 2.0
    db2 = db / dsum * 2.0
    res = img.copy()
    res[:,:,0] *= (db2 + dg) / 2.0
    res[:,:,1] *= dg
    res[:,:,2] *= db2
    return res

def recover_image(img, depths, B, beta_D, nmap):
    # img: HxWx3 float [0..1]
    # depths: HxW
    res = (img - B) * np.exp(beta_D * np.expand_dims(depths, axis=2))
    res = np.clip(res, 0.0, 1.0)
    res[nmap == 0] = 0
    try:
        res = scale01(wbalance_no_red_10p(res))
    except Exception:
        res = scale01(res)
    res[nmap == 0] = img[nmap == 0]
    return res

# ----------------------------
# Top-level function
# ----------------------------
def run_seathru_pipeline(img_input_uint8, depths, save_intermediate=False,
                         params=None):
    """
    img_input_uint8: HxWx3 uint8 (0..255) or float [0..1]
    depths: HxW float (same spatial size as img_input)
    params: optional dict to override algorithm parameters:
        {
            'min_depth_percent': 0.0,
            'backscatter_bins': 10,
            'backscatter_fraction': 0.01,
            'spread_data_fraction': 0.01,
            'p': 0.01,
            'f': 2.0,
            'l': 0.5
        }
    Returns recovered image as uint8 HxWx3
    """
    start = time.time()
    if params is None:
        params = {}
    p_local = params.get('p', 0.01)
    f_local = params.get('f', 2.0)
    l_local = params.get('l', 0.5)
    min_depth_percent = params.get('min_depth_percent', 0.0)
    backscatter_bins = params.get('backscatter_bins', 10)
    backscatter_fraction = params.get('backscatter_fraction', 0.01)
    spread_data_fraction = params.get('spread_data_fraction', 0.01)

    # normalize input image to float [0..1]
    if img_input_uint8.dtype == np.uint8:
        img = (img_input_uint8.astype(np.float32) / 255.0)
    else:
        img = img_input_uint8.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0

    depths = depths.astype(np.float32)
    # ensure non-zero depths handled
    depths[np.isnan(depths)] = 0.0

    # 1. backscatter estimation
    ptsR, ptsG, ptsB = find_backscatter_estimation_points(img, depths,
                                                         num_bins=backscatter_bins,
                                                         fraction=backscatter_fraction,
                                                         max_vals=20,
                                                         min_depth_percent=min_depth_percent)

    Br, coefsR = find_backscatter_values(ptsR, depths, restarts=25)
    Bg, coefsG = find_backscatter_values(ptsG, depths, restarts=25)
    Bb, coefsB = find_backscatter_values(ptsB, depths, restarts=25)

    B = np.stack([Br, Bg, Bb], axis=2)

    # 2. neighborhood map
    nmap, _ = construct_neighborhood_map(depths, epsilon=0.1)
    nmap_refined, n_labels = refine_neighborhood_map(nmap, min_size=50, radius=3)

    # 3. illumination estimates per channel
    illR = estimate_illumination(img[:,:,0], Br, nmap_refined, n_labels, p=p_local, f=f_local, max_iters=100, tol=1e-5)
    illG = estimate_illumination(img[:,:,1], Bg, nmap_refined, n_labels, p=p_local, f=f_local, max_iters=100, tol=1e-5)
    illB = estimate_illumination(img[:,:,2], Bb, nmap_refined, n_labels, p=p_local, f=f_local, max_iters=100, tol=1e-5)
    ill = np.stack([illR, illG, illB], axis=2)

    # 4. estimate attenuation beta_D initial and refine
    beta_D_r, _ = estimate_wideband_attentuation(depths, illR)
    refined_beta_D_r, coefsR = refine_wideband_attentuation(depths, illR, beta_D_r, radius_fraction=spread_data_fraction, l=l_local)
    beta_D_g, _ = estimate_wideband_attentuation(depths, illG)
    refined_beta_D_g, coefsG = refine_wideband_attentuation(depths, illG, beta_D_g, radius_fraction=spread_data_fraction, l=l_local)
    beta_D_b, _ = estimate_wideband_attentuation(depths, illB)
    refined_beta_D_b, coefsB = refine_wideband_attentuation(depths, illB, beta_D_b, radius_fraction=spread_data_fraction, l=l_local)

    beta_D = np.stack([refined_beta_D_r, refined_beta_D_g, refined_beta_D_b], axis=2)

    # 5. reconstruct image
    recovered = recover_image(img, depths, B, beta_D, nmap_refined)

    # optional post-processing: tv denoise with skimage wrapper
    try:
        sigma_est = _estimate_sigma_compat(recovered, channel_axis=-1, average_sigmas=True) / 10.0
        recovered = _denoise_tv_chambolle_compat(recovered, sigma_est, channel_axis=-1)
    except Exception:
        pass

    # convert to uint8 0..255
    recovered_uint8 = (np.clip(recovered, 0.0, 1.0) * 255.0).astype(np.uint8)

    # save intermediates if requested
    if save_intermediate:
        try:
            Image.fromarray((img*255).astype(np.uint8)).save("debug_input.png")
            Image.fromarray((B * 255).astype(np.uint8)).save("debug_backscatter.png")
            Image.fromarray((ill * 255).astype(np.uint8)).save("debug_illum.png")
            Image.fromarray((scale01(beta_D[:,:,0]) * 255).astype(np.uint8)).save("debug_betaD_r.png")
            Image.fromarray(recovered_uint8).save("debug_recovered.png")
        except Exception:
            pass

    # done
    return recovered_uint8

# If run as script for ad-hoc testing (not necessary for library usage)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth", required=True, help="depth numpy .npy file (H,W) or image")
    parser.add_argument("--output", default="out_seathru.png")
    parser.add_argument("--save-intermediate", action="store_true")
    args = parser.parse_args()
    # load
    img = np.array(Image.open(args.image).convert("RGB"))
    # depth can be .npy or grayscale image
    depth_path = args.depth
    if depth_path.endswith(".npy"):
        depths = np.load(depth_path)
    else:
        depth_img = Image.open(depth_path).convert("L")
        depths = np.array(depth_img).astype(np.float32) / 255.0
    out = run_seathru_pipeline(img, depths, save_intermediate=args.save_intermediate)
    Image.fromarray(out).save(args.output)
    print("Saved", args.output)
