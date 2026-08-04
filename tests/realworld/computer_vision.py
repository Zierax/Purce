"""Computer vision patterns: NMS, RoI pooling, spatial transforms."""

import numpy as np


def intersection_over_union(box1, box2):
    x1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[2], box2[2])
    y2 = np.minimum(box1[3], box2[3])
    intersection = np.maximum(0, np.multiply(np.subtract(x2, x1), np.subtract(y2, y1)))
    area1 = np.multiply(np.subtract(box1[2], box1[0]), np.subtract(box1[3], box1[1]))
    area2 = np.multiply(np.subtract(box2[2], box2[0]), np.subtract(box2[3], box2[1]))
    union = np.subtract(np.add(area1, area2), intersection)
    return np.divide(intersection, np.add(union, 1e-8))


def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    order = np.argsort(np.negative(scores))
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        remaining = order[1:]
        ious = np.zeros(len(remaining))
        for j_idx, j in enumerate(remaining):
            ious[j_idx] = intersection_over_union(boxes[i], boxes[j])
        mask = np.less(ious, iou_threshold)
        order = remaining[mask]
    return keep


def spatial_transform(features, theta):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return np.matmul(features, rotation)


def bilinear_interpolate(feature_map, x, y):
    h, w = feature_map.shape
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, h - 1)
    y1 = min(y0 + 1, w - 1)
    dx = np.subtract(x, x0)
    dy = np.subtract(y, y0)
    val = np.add(
        np.add(np.multiply(np.multiply(np.subtract(1.0, dx), np.subtract(1.0, dy)), feature_map[x0, y0]),
               np.multiply(np.multiply(dx, np.subtract(1.0, dy)), feature_map[x1, y0])),
        np.add(np.multiply(np.multiply(np.subtract(1.0, dx), dy), feature_map[x0, y1]),
               np.multiply(np.multiply(dx, dy), feature_map[x1, y1]))
    )
    return val


def roi_pooling(feature_map, roi, output_size):
    h = output_size
    w = output_size
    roi_h = np.subtract(roi[2], roi[0])
    roi_w = np.subtract(roi[3], roi[1])
    bin_h = np.divide(roi_h, h)
    bin_w = np.divide(roi_w, w)
    pooled = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            y_start = int(np.add(roi[0], np.multiply(i, bin_h)))
            x_start = int(np.add(roi[1], np.multiply(j, bin_w)))
            y_end = int(np.add(roi[0], np.multiply(np.add(i, 1), bin_h)))
            x_end = int(np.add(roi[1], np.multiply(np.add(j, 1), bin_w)))
            y_start = max(0, min(y_start, feature_map.shape[0] - 1))
            x_start = max(0, min(x_start, feature_map.shape[1] - 1))
            y_end = max(y_start + 1, min(y_end, feature_map.shape[0]))
            x_end = max(x_start + 1, min(x_end, feature_map.shape[1]))
            pooled[i, j] = np.max(feature_map[y_start:y_end, x_start:x_end])
    return pooled


def deformable_conv_offset(features, kernel_size):
    h, w = features.shape
    offsets = np.zeros((kernel_size, kernel_size, 2))
    for i in range(kernel_size):
        for j in range(kernel_size):
            offsets[i, j, 0] = np.mean(features)
            offsets[i, j, 1] = np.mean(features)
    return offsets


def non_local_block(x, theta_channels, phi_channels, g_channels):
    batch = x.shape[0]
    theta = np.matmul(x.reshape(batch, -1), theta_channels)
    phi = np.matmul(x.reshape(batch, -1), phi_channels)
    g = np.matmul(x.reshape(batch, -1), g_channels)
    scores = np.matmul(theta, np.transpose(phi))
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    out = np.matmul(weights, g)
    return out.reshape(x.shape)
