import heapq
import math
import time

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
from sympy import Line, Point


threshold = 1.5
rhoRes = 2
thetaRes = math.pi/180
resizeWidth = 840


def time_since(s):
    ''' 
    Returns time since parameter s rounded to 1/1000 of a second
    '''
    return round((time.time() - s) * 1000) / 1000


def save(img, name):
    Image.fromarray(np.uint8(img*255)).save(f'{name}.png')
    

def find_intersections(line, k, lines):
    found = []

    for i in range(len(lines)):
        if i != k:
            rho, theta = lines[i][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            corners = Line(Point(x1, y1), Point(x2, y2)).intersection(line)
            for c in corners:
                found.append(c.coordinates)

    return found


def detect_field(path, saveImg=False):
    # Load image
    img = Image.open(path).convert('RGB')

    # Resize
    w, h = img.size
    scale = resizeWidth / w
    img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
    colImg = np.array(img)

    img = img.convert("L")
    # Normalize and suppress dark regions
    igs = np.array(img) / 255.
    igs = np.maximum(igs, 0.5)

    if saveImg:
        save(igs, "01-grayscale")

    print("Detect edges...")
    s = time.time()
    Im = cv.Canny(np.uint8(igs*255), 100, 200, apertureSize=3) / 255.
    if saveImg:
        save(Im, "02-edges")
    print("Edges took", time_since(s))

    print("Hough transform...")
    s = time.time()
    lines = cv.HoughLines(np.uint8(Im*255), rhoRes, thetaRes, int(threshold*255))
    print("CV hough lines took", time_since(s))

    draw = ImageDraw.Draw(img)

    # Todo: only compare with promising candidates
    # (ie rule out lines with same orientation or in the middle of the field)
    print("Find intersections...")
    s = time.time()
    myCorners = []
    for i in range(len(lines)):
        rho, theta = lines[i][0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        # x1,y1,x2,y2 = line[0]
        draw.line(((x1, y1), (x2, y2)), fill="white")
        myCorners = myCorners + find_intersections(
            Line(Point(x1, y1), Point(x2, y2)), i, lines)

    tl = (-1, -1)
    tr = (-1, -1)
    bl = (-1, -1)
    br = (-1, -1)

    def closerToCenter(first, second):
        x1, y1 = first
        x2, y2 = second

        mx, my = resizeWidth/2, h*resizeWidth/w/2
        d1 = (mx - x1)**2 + (my - y1)**2
        d2 = (mx - x2)**2 + (my - y2)**2

        # It should not be in the middle
        middle = (mx - resizeWidth/2)**2 + (my - h*resizeWidth/w)**2
        if (d2 < middle):
            return first
        return first if d1 < d2 else second

    for i in range(len(myCorners)):
        x, y = myCorners[i]
        if 0 < x < resizeWidth/2 and 0 < y < h*resizeWidth/w/2:
            tl = closerToCenter(tl, (x, y))
        elif resizeWidth/2 < x < resizeWidth and 0 < y < h*resizeWidth/w/2:
            tr = closerToCenter(tr, (x, y))
        elif 0 < x < resizeWidth/2 and h*resizeWidth/w/2 < y < h*resizeWidth/w:
            bl = closerToCenter(bl, (x, y))
        elif resizeWidth/2 < x < resizeWidth and h*resizeWidth/w/2 < y < h*resizeWidth/w:
            br = closerToCenter(br, (x, y))

    print("Finding corners took", time_since(s))
    # Draw rectangle
    draw.line([tr, tl], fill="black")
    draw.line([tl, bl], fill="black")
    draw.line([bl, br], fill="black")
    draw.line([br, tr], fill="black")

    if saveImg:
        Image.fromarray(np.uint8(img)).save(f'04-img-with-hough-lines.png')


    return colImg, (tl, tr, bl, br), scale


def compute_h(p1, p2):
    x, y = 0, 1
    p = np.empty((len(p1)*2, 9))

    for r in range(len(p1)):
        p[2*r] = np.array([
            [p2[r, x], p2[r, y], 1, 0, 0, 0, -p1[r, x] *
                p2[r, x], -p1[r, x]*p2[r, y], -p1[r, x]]
        ])
        p[2*r+1] = np.array([
            [0, 0, 0, p2[r, x], p2[r, y], 1, -p1[r, y] *
                p2[r, x], -p1[r, y]*p2[r, y], -p1[r, y]]
        ])

    # With help from https://math.stackexchange.com/a/3511513
    _, _, Vt = np.linalg.svd(p)
    H = Vt[-1].reshape(3, 3)

    return H


def warp_image(igs_in, igs_ref, H):
    # img in: (row, col) = (y, x) = (h, w)
    in_h, in_w, d = igs_in.shape
    ref_h, ref_w, _ = igs_ref.shape

    print("Dimensions", in_h, in_w, ref_h, ref_w)

    # calculate outer bounderies
    warped_edges = (H @ np.array([
        [0, 0, 1],
        [in_w, 0, 1],
        [0, in_h, 1],
        [in_w, in_h, 1]
    ]).T)
    warped_edges = warped_edges/warped_edges[-1]

    offset_x = -int(min(0, warped_edges[0].min()))
    offset_y = -int(min(0, warped_edges[1].min()))
    mrg_w = int(max(ref_w, warped_edges[0].max())) + offset_x
    mrg_h = int(max(ref_h, warped_edges[1].max())) + offset_y

    # init images
    igs_warp = np.zeros((ref_h, ref_w, d))
    igs_merge = np.zeros((mrg_h, mrg_w, d))

    # warp into ref and merge result
    for x in range(mrg_w):
        for y in range(mrg_h):
            r = np.linalg.inv(H) @ np.array([x-offset_x, y-offset_y, 1]).T
            xx, yy, _ = r/r[-1]
            i, j = int(xx), int(yy)
            a, b = xx-i, yy-j

            if 0 <= i < in_w-1 and 0 <= j < in_h-1:
                c = (1-a)*(1-b) * igs_in[j, i] \
                    + a*(1-b) * igs_in[j, i+1] \
                    + a*b * igs_in[j+1, i+1] \
                    + (1-a)*b * igs_in[j+1, i]

                igs_merge[y, x] = c

                if 0 <= x-offset_x < ref_w and 0 <= y-offset_y < ref_h:
                    igs_warp[y-offset_y, x-offset_x] = c

    # merge
    igs_merge[offset_y:ref_h+offset_y, offset_x:ref_w+offset_x] = igs_ref[:]

    return igs_warp, igs_merge


def rectify(igs, p1, p2):
    H = compute_h(p1, p2)
    igs_rec, _ = warp_image(igs, np.zeros((550, 1050, 3)), H)
    return igs_rec, H


def transform_image(igs, corners):
    c_in = np.array([
        [25, 25],
        [1025, 25],
        [25, 525],
        [1025, 525]
    ])
    
    c_ref = np.int32(np.array(corners))
    igs_rec, H = rectify(np.array(igs), c_in, c_ref)

    Image.fromarray(np.uint8(igs_rec)).save(f'05-warped.png')

    return igs_rec, H


def main():
    igs, corners, scale = detect_field('static.png', True)
    img, H = transform_image(igs, corners)


if __name__ == '__main__':
    main()
