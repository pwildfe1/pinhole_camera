import os
import logging
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

from scipy import interpolate
from scipy.spatial.transform import Rotation


class HandPointCloud:

    def __init__(self, pcd, keypoints, positions):

        self.pc = pcd
        self.pc.estimate_normals()
        self.v = np.asarray(self.pc.points)
        self.vn = np.asarray(self.pc.normals)
        self.vc = np.asarray(self.pc.colors)
        self.pos = positions
        self.markers = keypoints

        self.origin = []
        self.z_axis = []

        self.fingers = []
        self.fingers_indices = [[2, 3, 4, 5]]
        self.fingers_indices.append([6, 7, 8, 9])
        self.fingers_indices.append([9, 10, 11, 12])
        self.fingers_indices.append([13, 14, 15, 16])
        self.fingers_indices.append([17, 18, 19, 20])

        keyindices = np.array(self.markers, dtype=int)
        self.markervertices = self.pos[keyindices[:, 1], keyindices[:, 0]]
        for i in range(len(self.fingers_indices)):
            self.fingers.append(extract_finger(self.pc, self.markervertices, self.fingers_indices[i]))




    def extract_table(self):

        baseline = []
        baseline_normal = []

        for i in range(len(self.fingers)):

            f_indices = self.fingers_indices[i]

            table_attractor = self.markervertices[f_indices[-1]] + (self.markervertices[f_indices[-1]] - self.markervertices[f_indices[-2]])

            vecs_frame = self.v[:,0:2] - table_attractor[0:2]
            dists_frame = np.linalg.norm(vecs_frame, axis = 1)
            table_attractor = self.v[np.argmin(dists_frame)]

            vecs = self.v[:] - table_attractor
            dists = np.linalg.norm(vecs, axis = 1)
            closest = np.argsort(dists)[0:30]
            finger_normal = np.mean(self.vn[closest], axis = 0)
            finger_normal = finger_normal/np.linalg.norm(finger_normal)

            z_values = np.dot(self.v[closest], finger_normal)
            z_values = z_values - np.min(z_values)
            finger_baseline = self.v[closest][np.argmin(z_values)] + np.median(z_values) * finger_normal

            print(finger_baseline)

            baseline.append(finger_baseline)
            baseline_normal.append(finger_normal)

        self.origin = np.mean(np.array(baseline), axis = 0)
        self.z_axis = np.mean(np.array(baseline_normal), axis = 0)
        self.z_axis = self.z_axis/np.linalg.norm(self.z_axis)

        print(self.z_axis)

        vecs = self.v[:] - self.origin
        z = np.abs(np.dot(vecs, self.z_axis))

        print(z)

        table_indices = np.where(z < .5)[0]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.v[table_indices])
        pc.colors = o3d.utility.Vector3dVector(self.vc[table_indices])

        o3d.visualization.draw_geometries([pc])




def extract_finger(pcd, keyvertices, indices, thres = 3):

    keyvertices = keyvertices[indices]
    v = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    pts = []

    for i in range(keyvertices.shape[0] - 1):
        
        vec = keyvertices[i + 1] - keyvertices[i]
        steps = int(np.linalg.norm(vec)/thres)

        for j in range(steps):
            pts.append(keyvertices[i] + vec/np.linalg.norm(vec) * thres * j)

    pts = np.array(pts)

    for i in range(pts.shape[0]):
        key_flat = pts[i,0:2]
        v_flat = v[:, 0:2]
        vecs = v_flat - key_flat
        dist = np.linalg.norm(vecs, axis = 1)

        if i == 0:
            indices = np.where(dist < 1)[0]
        else:
            indices = np.concatenate((indices, np.where(dist < thres)[0]), axis = 0)
        print(indices.shape)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(v[indices])
    pc.colors = o3d.utility.Vector3dVector(colors[indices])

    return pc



def main(pc_file, positions_file):
    
    pcd = o3d.io.read_point_cloud(pc_file)
    pcd = pcd.scale(100, [0, 0, 0])

    positions = np.load(positions_file)

    f = open("results/results_single_frame.json", 'r')
    result = json.load(f)
    keypoints = result["instance_info"][0]["keypoints"]

    myHand = HandPointCloud(pcd, keypoints, positions)
    myHand.extract_table()


    # o3d.visualization.draw_geometries([myHand.fingers[0], myHand.fingers[1], myHand.fingers[2]])

    # g = open("results/isolated.xyz", "w")
    # v = np.asarray(thumb_pc.points)

    # g.write("# Thumb\n#\n")
    # for i in range(v.shape[0]):
    #     for j in range(v.shape[1]):
    #         g.write(str(v[i, 0]))
    #         if j < v.shape[1] - 1: g.write(" ")
    #     if i < v.shape[0] - 1: g.write("\n")
    # g.close()



main("models/pc.ply", "saved_positions.npy")