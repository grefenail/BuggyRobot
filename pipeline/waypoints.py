"""Coordinate conversion helpers for lane center waypoints."""

import math

import cv2
import numpy as np

import config
from ipm import get_matrices


def transform_point(matrix, point):
    pt = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, matrix)[0][0]
    return float(out[0]), float(out[1])


def pixel_to_ground_m(
    pixel,
    width,
    height,
    fx=config.APPROX_CAMERA_FX_PX,
    fy=config.APPROX_CAMERA_FY_PX,
    camera_height_m=config.APPROX_CAMERA_HEIGHT_M,
    pitch_deg=config.APPROX_CAMERA_PITCH_DEG,
):
    """
    Project a vehicle-view image pixel onto the ground plane using a
    simple pinhole camera and a front-facing, downward-pitched camera.

    Robot frame: x = forward meters, y = left meters, z = up.
    """
    cx = width / 2.0
    cy = height / 2.0
    u, v = pixel

    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = 1.0

    ray_level = np.array([z_cam, -x_cam, -y_cam], dtype=np.float64)
    pitch = math.radians(pitch_deg)
    rot_y = np.array([
        [math.cos(pitch), 0.0, math.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-math.sin(pitch), 0.0, math.cos(pitch)],
    ])
    ray_robot = rot_y @ ray_level

    if ray_robot[2] >= -1e-6:
        return None

    scale = camera_height_m / -ray_robot[2]
    ground = ray_robot * scale
    return {
        "x_forward_m": round(float(ground[0]), 4),
        "y_left_m": round(float(ground[1]), 4),
    }


def add_approx_ground_waypoints(
    coords,
    fx=config.APPROX_CAMERA_FX_PX,
    fy=config.APPROX_CAMERA_FY_PX,
    camera_height_m=config.APPROX_CAMERA_HEIGHT_M,
    pitch_deg=config.APPROX_CAMERA_PITCH_DEG,
):
    waypoints_px = coords.get("center_waypoints_px")
    if not waypoints_px:
        coords["center_waypoints_m_approx"] = None
        return coords

    width = coords["bird_width"]
    height = coords["bird_height"]
    process_scale = float(coords.get("process_scale", 1.0))
    fx *= process_scale
    fy *= process_scale
    _, bird_to_vehicle = get_matrices(width, height)

    waypoints_m = []
    for bird_px in waypoints_px:
        vehicle_px = transform_point(bird_to_vehicle, bird_px)
        ground = pixel_to_ground_m(
            vehicle_px, width, height, fx, fy, camera_height_m, pitch_deg
        )
        waypoints_m.append({
            "bird_px": [int(bird_px[0]), int(bird_px[1])],
            "vehicle_px": [round(vehicle_px[0], 2), round(vehicle_px[1], 2)],
            "ground_m": ground,
        })

    coords["center_waypoints_m_approx"] = waypoints_m
    return coords
