# warp/triangulation.py

import cv2

def delaunay_triangulation(points, image_shape):
    h, w = image_shape[:2]
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))

    triangles = subdiv.getTriangleList()
    tri_indices = []

    for t in triangles:
        tri = [(int(t[0]), int(t[1])),
               (int(t[2]), int(t[3])),
               (int(t[4]), int(t[5]))]

        idx = []
        for pt in tri:
            for i, p in enumerate(points):
                if abs(pt[0] - p[0]) < 1 and abs(pt[1] - p[1]) < 1:
                    idx.append(i)
        if len(idx) == 3:
            tri_indices.append(idx)

    return tri_indices
