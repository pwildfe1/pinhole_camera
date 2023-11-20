import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

from scipy import interpolate
from scipy.spatial.transform import Rotation


class PinholeCamera:

    def __init__(self, fx, fy, px = 0, py = 0):
        self.fx, self.fy = fx, fy
        self.px, self.py = px, py
        self.K = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
        self.I = np.identity(3)
        self.I = np.vstack((self.I, np.zeros((3)))).T
        self.P = np.matmul(self.K, self.I)

    def setCameraTransform(self, position, angles):
        self.I = Rotation.from_euler('xyz', angles, degrees = True).as_matrix()
        position = np.array(position)
        self.I = np.vstack((self.I, position)).T
        self.P = np.matmul(self.K, self.I)
        print(self.I)
        print(self.P)

    def renderPts(self, pts, colors):
        px = []
        for v in pts:
            pt = np.zeros((4,1))
            pt[0] = v[0]
            pt[1] = v[1]
            pt[2] = v[2]
            pt[3] = 1
            homogenous = np.matmul(self.P,pt)
            px.append([self.fx * homogenous[0]/homogenous[2] + self.px, self.fy * homogenous[1]/homogenous[2] + self.py])
        px = np.array(px)
        px = px[:, :, 0]
        print(px.shape)
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        ax.scatter(px[:,0], px[:,1], color=colors, s=1)
        ax.set_ylim(-540, 540)
        ax.set_xlim(-540, 540)
        # plt.show()

        return px


class Snapshot:

    def __init__(self, v, px, colors, width = 1080, height = 1080):

        self.v = v
        self.px = px
        self.colors = colors
        self.width = width
        self.height = height

        self.pixels = np.zeros(self.px.shape)
        self.pixels[:] = self.px[:]
        self.pixels = self.pixels.astype(int)

        print(self.pixels)


    def evaluate_pixel(self, pixel):

        index = -1
        pt = np.array([])

        index_u = np.where(self.pixels[:, 0] == pixel[0])[0]
        index_v = np.where(self.pixels[:, 1] == pixel[1])[0]

        if len(list(index_v)) == 0 or len(list(index_v)) == 0:
            return [index, pt]

        index_of_indices = np.where(index_u - index_v[0] == 0)[0]
        index_of_indices = list(index_of_indices)

        if len(index_of_indices) > 0:
            index = index_u[index_of_indices[0]]
            pt = self.v[index]

        return [index, pt]


    def export_image(self):

        data = np.zeros((self.width, self.height, 3))
        data_indices = np.zeros((self.width, self.height))
        pixels = self.pixels[:] + int(self.width/2)

        target_pixels, indices = np.unique(pixels, axis = 0, return_index=True)

        for i in range(target_pixels.shape[0]):
            if target_pixels[i, 0] < 1080 and target_pixels[i, 0] > 0 and target_pixels[i, 1] < 1080 and target_pixels[i, 1] > 0:
                data[target_pixels[i, 0], target_pixels[i, 1]] = self.colors[indices[i]]
                data_indices[target_pixels[i, 0], target_pixels[i, 1]] = indices[i]


        data = data[:]*255
        im = data.astype("uint8")
 
        img = Image.fromarray(im)
        img.save("single_frame.png")
        np.save("saved_positions.npy", data_indices)







def main(file):
    pcd = o3d.io.read_point_cloud(file)
    pcd = pcd.scale(100, [0, 0, 0])
    v = np.asarray(pcd.points)
    pcd = pcd.translate([0, 0, -(np.max(v[:,2]) - np.min(v[:,2]))])
    # pcd = pcd.voxel_down_sample(voxel_size=0.05)
    # search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.001, max_nn=30)
    myCamera = PinholeCamera(35, 35)
    myCamera.setCameraTransform([0, 0, 0], [0, 0, 0])

    v_colors = np.asarray(pcd.colors)

    v = np.asarray(pcd.points)
    projected = myCamera.renderPts(v, v_colors)
    print(v.shape)
    print(projected.shape)
    mySnapshot = Snapshot(v, projected, v_colors, width = 1080, height = 1080)
    mySnapshot.export_image()

    # o3d.visualization.draw_geometries([pcd])


main("models/pc.ply")