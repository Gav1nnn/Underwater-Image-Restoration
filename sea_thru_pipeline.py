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
# skimage API 变更的兼容性包装器
from skimage.restoration import denoise_tv_chambolle as _denoise_tv_chambolle
from skimage.restoration import estimate_sigma as _estimate_sigma

# ----------------------------
# 兼容性辅助函数
# ----------------------------

# 兼容 skimage API 变更的 estimate_sigma 包装器
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

# 兼容 skimage API 变更的 denoise_tv_chambolle 包装器
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
# 最小数值型辅助函数
# ----------------------------
_eps = 1e-8 # 避免除以零或对数运算的数值稳定小量

# 将数组归一化
def scale01(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < _eps:
        return np.clip(arr - mn, 0.0, 1.0)
    return (arr - mn) / (mx - mn)

# ----------------------------
# Sea-Thru 构成块
# ----------------------------

def find_backscatter_estimation_points(img, depths, num_bins=10, fraction=0.01, max_vals=20, min_depth_percent=0.0):
    """
    根据深度对图像进行分箱，并在每个深度范围内选择最暗（反向散射最小）的像素点作为估计点。
    这些点用于估计反向散射模型 B(z)。

    img: HxWx3 float [0..1] 图像
    depths: HxW float 深度图 (米或相对值)
    returns 形状为 (N,2) 的数组列表: (深度, 像素值)
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
    对估计点B_{pts}进行非线性拟合，以获得反向散射图 B(z)。
    拟合模型 (Sea-Thru 论文中的完整水下成像模型):
    B(z) = B_inf * (1 - exp(-beta_B * z)) + J' * exp(-beta_D' * z)
    其中 beta_{D} 是一个经验衰减系数 (并非真正的 beta_D)。
    如果非线性拟合失败或损失过大，则回退到线性拟合。

    B_pts: (N,2) 形状的数组 (深度, 像素值)
    depths: HxW 深度图
    """
    if B_pts.shape[0] == 0:
        # 如果没有点，返回零图和零系数
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
            # 忽略失败的重启
            pass
    if coefs is None or best_loss > max_mean_loss:
        # 线性拟合回退
        try:
            slope, intercept, *_ = sp.stats.linregress(B_depths, B_vals)
            BD = (slope * depths) + intercept
            return BD, np.array([slope, intercept])
        except Exception:
            return np.zeros_like(depths), np.array([0.,0.])
    return estimate(depths, *coefs), coefs

def construct_neighborhood_map(depths, epsilon=0.05):
    """
    基于深度相似性进行区域生长 (Flood-fill) 邻域分组。
    用于在局部空间上估计照明分量。

    depths: HxW 深度图
    epsilon: 深度相似性阈值 (占深度范围的百分比)
    Returns nmap (H,W) 带有标签 (0 表示背景) 和 num_labels (标签数量)
    """
    eps = (np.nanmax(depths) - np.nanmin(depths)) * epsilon
    h,w = depths.shape
    nmap = np.zeros_like(depths, dtype=np.int32)
    label = 1 # 区域标签
    unassigned = np.where(nmap==0)
    # 迭代区域生长
    for i in range(h):
        for j in range(w):
            if nmap[i,j] != 0:
                continue
            # 开始区域生长
            stack = [(i,j)]
            while stack:
                x,y = stack.pop()
                if not (0<=x<h and 0<=y<w):
                    continue
                if nmap[x,y] != 0:
                    continue
                if abs(depths[x,y] - depths[i,j]) <= eps:
                    nmap[x,y] = label
                    # 领域
                    if x+1 < h: stack.append((x+1,y))
                    if x-1 >= 0: stack.append((x-1,y))
                    if y+1 < w: stack.append((x,y+1))
                    if y-1 >= 0: stack.append((x,y-1))
            label += 1
    # 将最大的深度为 0 的区域 (背景/无效深度) 设置为标签 0
    uniq, counts = np.unique(nmap[depths==0], return_counts=True)
    if uniq.size > 0:
        largest_label = uniq[np.argmax(counts)]
        nmap[nmap==largest_label] = 0
    return nmap, label-1

def find_closest_label(nmap, sx, sy):
    """
    使用广度优先搜索 (BFS) 找到最近的非零标签。
    用于将小区域分配给最近的大区域。
    """
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
    """
    精炼邻域地图：
    1. 移除小于 min_size 的小区域。
    2. 将小区域分配给最近的大区域。
    3. 应用形态学闭运算 (Closing) 填充小孔。
    """
    refined = np.zeros_like(nmap)
    vals, counts = np.unique(nmap, return_counts=True)
    # 按照大小降序排序
    order = np.argsort(-counts)
    label_id = 1
    # 仅保留大区域
    for idx in order:
        val = vals[idx]
        cnt = counts[idx]
        if val == 0:
            continue
        if cnt >= min_size:
            refined[nmap==val] = label_id
            label_id += 1
    # 将小区域分配给最近的大区域
    for idx in order:
        val = vals[idx]
        cnt = counts[idx]
        if val == 0 or cnt >= min_size:
            continue
        coords = np.column_stack(np.where(nmap==val))
        for (x,y) in coords:
            refined[x,y] = find_closest_label(refined, x, y)
    # 闭运算填充小孔和缝隙
    refined = closing(refined, square(radius))
    return refined, label_id-1

def estimate_illumination(img_channel, B_channel, neighborhood_map, num_neighborhoods, p=0.5, f=2.0, max_iters=100, tol=1e-5):
    """
    基于邻域平均的迭代方法估计局部照明图 $C(x,y)$ (每个通道独立进行)。
    这是 Sea-Thru 算法的核心步骤之一。

    img_channel: HxW 图像通道 (例如 R)
    B_channel: HxW 反向散射通道
    neighborhood_map: HxW 邻域地图 (标签 1..N)
    num_neighborhoods: 标签数量
    p: 局部照明的权重
    f: 最终照明图的比例因子 (用于白平衡)
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
            # 计算邻域内除自身外的平均局部照明
            s = np.sum(avg_cs[locs]) - avg_cs[locs]
            avg_cs_prime[locs] = s / max(size,1)
        new_avg_cs = (D * p) + (avg_cs_prime * (1 - p))
        if np.max(np.abs(avg_cs - new_avg_cs)) < tol:
            avg_cs = new_avg_cs
            break
        avg_cs = new_avg_cs
    # 双边滤波去噪并乘以比例因子 f (illumination)
    # f 是一个白平衡相关的比例因子
    illum = f * denoise_bilateral(np.maximum(0, avg_cs))
    return illum

def estimate_wideband_attentuation(depths, illum, radius=6, max_val=10.0):
    """
    从照明图和深度图的初始估计中粗略估计宽带衰减系数 $\beta_D$。
    """
    eps = 1E-8
    BD = np.minimum(max_val, -np.log(illum + eps) / (np.maximum(0, depths) + eps))
    mask = (depths > eps) & (illum > eps)
    refined = denoise_bilateral(closing(np.maximum(0, BD * mask), disk(radius)))
    return refined, []

def calculate_beta_D(depths, a, b, c, d):
    """
    $\beta_D$ 的非线性拟合模型 (经验衰减模型，双指数衰减):
    """
    return (a * np.exp(b * depths)) + (c * np.exp(d * depths))

def filter_data(X, Y, radius_fraction=0.01):
    """
    对 (X, Y) 数据点进行过滤和降采样，用于 $\beta_D$ 的拟合。
    它沿着 X 轴 (深度) 分割数据，并在每个区间内选取 Y 值 ( $\beta_D$ 估计值) 的中位数。
    这有助于减少异常值和数据量。
    """
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
    """
    对 $\beta_D$ 的初始估计进行非线性拟合，以获得更精确的 $\beta_D$ 图。
    这次拟合是基于图像重建损失 (而非拟合 $z$ vs $\beta_D$)。

    depths: HxW 深度图 z
    illum: HxW 照明图 C
    estimation: HxW $\beta_D$ 的初始估计值
    l: 比例因子 (用于白平衡)
    """
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
        # 线性拟合回退
        try:
            slope, intercept, *_ = sp.stats.linregress(depths[locs_mask], estimation[locs_mask])
            BD = l * (slope * depths + intercept)
            return BD, np.array([slope, intercept])
        except Exception:
            return estimation, np.array([0,0])
    BD = l * calculate_beta_D(depths, *coefs)
    return BD, coefs

def wbalance_no_red_10p(img):
    """
    基于灰色世界假设的白平衡，但忽略红色通道 (因为红光在水下衰减最快)。
    使用图像中最亮的 10% (绿/蓝) 
    """
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
    """
    使用反向散射 $B$ 和衰减系数 $\beta_D$ 重建图像 $J$。

    水下成像模型的逆运算:
    $J = (I - B) \cdot \exp(\beta_D \cdot z)$

    img: HxWx3 原始图像 I
    depths: HxW 深度图 z
    B: HxWx3 反向散射图 B
    beta_D: HxWx3 衰减系数图 $\beta_D$
    nmap: HxW 邻域地图 (用于处理背景/无效区域)
    """
    # 1. 恢复直接透射分量 J
    # 注意：np.expand_dims(depths, axis=2) 将 depths 扩展为 HxWx1，以进行 HxWx3 的乘法
    res = (img - B) * np.exp(beta_D * np.expand_dims(depths, axis=2))
    # 2. 裁剪到 [0, 1] 范围
    res = np.clip(res, 0.0, 1.0)
    # 3. 忽略背景 (nmap == 0) 的重建值，将其设置为 0
    res[nmap == 0] = 0
    # 4. 白平衡和归一化
    try:
        res = scale01(wbalance_no_red_10p(res))
    except Exception:
        res = scale01(res)
    # 5. 将背景 (nmap == 0) 恢复为原始图像 I 的对应像素值
    res[nmap == 0] = img[nmap == 0]
    return res

# ----------------------------
# 顶层函数
# ----------------------------
def run_seathru_pipeline(img_input_uint8, depths, save_intermediate=False,
                         params=None):
    """
    运行 Sea-Thru 水下图像恢复管道。

    img_input_uint8: HxWx3 uint8 (0..255) 或 float [0..1] 图像
    depths: HxW float 深度图
    params: 可选参数字典，用于覆盖默认算法参数。

    Returns 恢复后的图像 (uint8 HxWx3)
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

    # 归一化图像到 float [0..1]
    if img_input_uint8.dtype == np.uint8:
        img = (img_input_uint8.astype(np.float32) / 255.0)
    else:
        img = img_input_uint8.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0

    depths = depths.astype(np.float32)
    # 确保 NaN 深度被视为 0 (背景/无效深度)
    depths[np.isnan(depths)] = 0.0

    # 1. 反向散点估计（backscatter estimation）
    ptsR, ptsG, ptsB = find_backscatter_estimation_points(img, depths,
                                                         num_bins=backscatter_bins,
                                                         fraction=backscatter_fraction,
                                                         max_vals=20,
                                                         min_depth_percent=min_depth_percent)

    Br, coefsR = find_backscatter_values(ptsR, depths, restarts=25)
    Bg, coefsG = find_backscatter_values(ptsG, depths, restarts=25)
    Bb, coefsB = find_backscatter_values(ptsB, depths, restarts=25)

    B = np.stack([Br, Bg, Bb], axis=2)

    # 2. 邻域地图（neighborhood map）
    nmap, _ = construct_neighborhood_map(depths, epsilon=0.1)
    nmap_refined, n_labels = refine_neighborhood_map(nmap, min_size=50, radius=3)

    # 3. 照明估计 (Illumination estimates)
    illR = estimate_illumination(img[:,:,0], Br, nmap_refined, n_labels, p=p_local, f=f_local, max_iters=100, tol=1e-5)
    illG = estimate_illumination(img[:,:,1], Bg, nmap_refined, n_labels, p=p_local, f=f_local, max_iters=100, tol=1e-5)
    illB = estimate_illumination(img[:,:,2], Bb, nmap_refined, n_labels, p=p_local, f=f_local, max_iters=100, tol=1e-5)
    ill = np.stack([illR, illG, illB], axis=2)

    # 4. 衰减系数估计和精炼
    beta_D_r, _ = estimate_wideband_attentuation(depths, illR)
    refined_beta_D_r, coefsR = refine_wideband_attentuation(depths, illR, beta_D_r, radius_fraction=spread_data_fraction, l=l_local)
    beta_D_g, _ = estimate_wideband_attentuation(depths, illG)
    refined_beta_D_g, coefsG = refine_wideband_attentuation(depths, illG, beta_D_g, radius_fraction=spread_data_fraction, l=l_local)
    beta_D_b, _ = estimate_wideband_attentuation(depths, illB)
    refined_beta_D_b, coefsB = refine_wideband_attentuation(depths, illB, beta_D_b, radius_fraction=spread_data_fraction, l=l_local)

    beta_D = np.stack([refined_beta_D_r, refined_beta_D_g, refined_beta_D_b], axis=2)

    # 5. 重构图像
    recovered = recover_image(img, depths, B, beta_D, nmap_refined)

    # 可选后处理：TV (Total Variation) 去噪 (使用兼容性包装器)
    try:
        sigma_est = _estimate_sigma_compat(recovered, channel_axis=-1, average_sigmas=True) / 10.0
        recovered = _denoise_tv_chambolle_compat(recovered, sigma_est, channel_axis=-1)
    except Exception:
        pass

    # 转换回 uint8 0..255
    recovered_uint8 = (np.clip(recovered, 0.0, 1.0) * 255.0).astype(np.uint8)

    # 如果请求，保存中间结果
    if save_intermediate:
        try:
            Image.fromarray((img*255).astype(np.uint8)).save("debug_input.png")
            Image.fromarray((B * 255).astype(np.uint8)).save("debug_backscatter.png")
            Image.fromarray((ill * 255).astype(np.uint8)).save("debug_illum.png")
            Image.fromarray((scale01(beta_D[:,:,0]) * 255).astype(np.uint8)).save("debug_betaD_r.png")
            Image.fromarray(recovered_uint8).save("debug_recovered.png")
        except Exception:
            pass

    # 完成
    return recovered_uint8

# 作为脚本运行
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth", required=True, help="depth numpy .npy file (H,W) or image")
    parser.add_argument("--output", default="out_seathru.png")
    parser.add_argument("--save-intermediate", action="store_true")
    args = parser.parse_args()
    # 载入
    img = np.array(Image.open(args.image).convert("RGB"))
    # 载入深度图 (可以是 .npy 文件或灰度图像)
    depth_path = args.depth
    if depth_path.endswith(".npy"):
        depths = np.load(depth_path)
    else:
        depth_img = Image.open(depth_path).convert("L")
        # 归一化
        depths = np.array(depth_img).astype(np.float32) / 255.0
    # 运行 Sea- thru 管道
    out = run_seathru_pipeline(img, depths, save_intermediate=args.save_intermediate)
    # 保存结果
    Image.fromarray(out).save(args.output)
    print("Saved", args.output)
