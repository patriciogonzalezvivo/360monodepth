import os
import cv2
import argparse
import numpy as np

from utility.ply import write_ply


def coords2uv(coords, w, h):
    #output uv size w*h*2
    uv = np.zeros_like(coords, dtype = np.float32)
    middleX = w/2 + 0.5
    middleY = h/2 + 0.5
    uv[..., 0] = (coords[...,0] - middleX) / w * 2 * np.pi
    uv[..., 1] = -(coords[...,1] - middleY) / h * np.pi
    return uv


def uv2xyz(uv):
    xyz = np.zeros((uv.shape[0], 3), dtype = np.float32)
    xyz[:, 0] = np.multiply(np.cos(uv[:, 1]), np.sin(uv[:, 0]))
    xyz[:, 1] = np.multiply(np.cos(uv[:, 1]), np.cos(uv[:, 0]))
    xyz[:, 2] = np.sin(uv[:, 1])
    return xyz


def open_float_rgb(path):
    img = cv2.imread(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0


def rgb2hue(rgb):
    # convert numpy RGB image to 2D HUE image
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    hue = np.zeros_like(r)
    hue[r >= g] = (60.0 * (g[r >= g] - b[r >= g]) / (r[r >= g] - b[r >= g]) + 0.0) / 360.0
    hue[r < g] = (60.0 * (g[r < g] - b[r < g]) / (r[r < g] - b[r < g]) + 360.0) / 360.0
    hue[r < b] = (60.0 * (g[r < b] - b[r < b]) / (r[r < b] - b[r < b]) + 240.0) / 360.0 
    return hue 


def rgb2heat(rgb):
    return 1.025 - rgb2hue(rgb) * 1.538461538


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rgb', '-r', help="RGB input", type=str, required=True)
    parser.add_argument('-depth', '-d', help="depth input", type=str, required=True)
    args = parser.parse_args()

    rgb = cv2.imread(args.rgb)
    path_out = args.depth.split('.')[0] + ".ply"
    if args.depth.endswith(".npy"):
        depth = np.load(args.depth)
    else:
        depth_rgb = open_float_rgb(args.depth)
        depth = rgb2heat(depth_rgb)
    
    h, w, c = rgb.shape

    # save raw 3D point cloud reconstruction as ply file
    coords = np.stack(np.meshgrid(range(w), range(h)), -1)
    coords = np.reshape(coords, [-1, 2])
    coords += 1
    uv = coords2uv(coords, w, h)          
    xyz = uv2xyz(uv)

    rgb_buffer = np.reshape(rgb, (-1, 3)).astype(np.uint8)
    depth_buffer = np.reshape(depth, (-1, 1)).astype(np.float32)
    xyz_buffer = xyz * depth_buffer

    print("Output path:", path_out)
    write_ply(path_out, [xyz_buffer, rgb_buffer], ['x', 'y', 'z', 'blue', 'green', 'red'])