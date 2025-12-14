# warp/warp_triangle.py

import cv2
import numpy as np

def warp_triangle(img, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    src_rect = [(t_src[i][0]-r1[0], t_src[i][1]-r1[1]) for i in range(3)]
    dst_rect = [(t_dst[i][0]-r2[0], t_dst[i][1]-r2[1]) for i in range(3)]

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_rect), (1,1,1))

    img1 = img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    warp_mat = cv2.getAffineTransform(
        np.float32(src_rect), np.float32(dst_rect)
    )

    img2 = cv2.warpAffine(
        img1, warp_mat, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] *= (1 - mask)
    img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] += img2 * mask

    return img
