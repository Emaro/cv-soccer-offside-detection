import heapq
import math
import time

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
from sympy import Line, Point

sigma = 2.6
threshold = 1.5
rhoRes = 2
thetaRes = math.pi/180
nLines = 18
resizeWidth = 840


def SuppressNonMax(img, i, j, ang):
    dx, dy = round(math.cos(ang)), round(math.sin(ang))

    if (0 <= i+dx < len(img)
        and 0 <= j+dy < len(img[0])
            and img[i+dx, j+dy] > img[i, j]):
        return 0
    if (0 <= i-dx < len(img)
        and 0 <= j-dy < len(img[0])
            and img[i-dx, j-dy] > img[i, j]):
        return 0
    else:
        return img[i, j]

def timeSince(s):
    return round((time.time() - s) * 1000) / 1000

def EdgeDetection(Igs, sigma):
    s = time.time()
    Ims = cv.Canny(np.uint8(Igs*255), 100, 200, apertureSize=3) / 255.
    print("Canny took", timeSince(s))
    return Ims

def HoughTransform(Im, threshold, rhoRes, thetaRes):
    rhoMax = int(math.sqrt(len(Im)**2 + len(Im[0])**2))
    rhoLen = int(rhoMax / rhoRes)
    thetaMax = 2*math.pi
    thetaLen = int(thetaMax / thetaRes)

    H = np.zeros((rhoLen, thetaLen))

    for i in range(0, len(Im)):
        for j in range(0, len(Im[0])):
            Im[i, j] = 1 if Im[i, j] >= threshold else 0

    for i in range(0, len(Im)):
        for j in range(0, len(Im[0])):
            if Im[i, j] > 0:
                for t in range(0, thetaLen):
                    rho = int(i * np.cos(t*thetaRes) + j *
                              np.sin(t*thetaRes)) // rhoRes
                    H[rho, t] += 1

    H = H / H.max()

    return H


def HoughNonmax(img, i, j):
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if 0 <= i+dx < len(img) and 0 <= j+dy < len(img[0]) and img[i+dx, j+dy] > img[i, j]:
                return 0

    return img[i, j]


def HoughLines(H, rhoRes, thetaRes, nLines, no=0):
    lRho, lTheta = np.zeros(nLines), np.zeros(nLines)

    hs = np.zeros((len(H), len(H[0])))
    for i in range(len(H)):
        for j in range(len(H[0])):
            hs[i, j] = HoughNonmax(H, i, j)

    hp = [(0, 0, 0)] * nLines
    heapq.heapify(hp)

    for i in range(len(hs)):
        for j in range(len(hs[0])):
            if ((hs[i, j], 0, 0) > hp[0]):
                heapq.heapreplace(hp, (hs[i, j], i*rhoRes, j*thetaRes))

    hp = [(a, b, c) for (a, b, c) in hp if a > 0]
    _, lRho, lTheta = zip(*hp)
    return lRho, lTheta


def getLine(r, t):
    mag, rot = r, t - math.pi/2
    p1 = Point(+mag*math.cos(rot) - 1000*math.sin(rot),
               -mag*math.sin(rot) - 1000*math.cos(rot))
    p2 = Point(+mag*math.cos(rot) + 1000*math.sin(rot),
               -mag*math.sin(rot) + 1000*math.cos(rot))
    ln = Line(p1, p2)
    return ln


def findIntersections(line, k, lines):
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
            corners = Line(Point(x1,y1), Point(x2,y2)).intersection(line)
            for c in corners:
                found.append(c.coordinates)

    return found


def drawLine(draw, mag, rot, color="yellow"):
    draw.line((
        +mag*math.cos(rot),
        -mag*math.sin(rot),
        +mag*math.cos(rot) - 1000*math.sin(rot),
        -mag*math.sin(rot) - 1000*math.cos(rot)), fill=color)
    draw.line((
        +mag*math.cos(rot),
        -mag*math.sin(rot),
        +mag*math.cos(rot) + 1000*math.sin(rot),
        -mag*math.sin(rot) + 1000*math.cos(rot)), fill=color)


def save(img, name):
    Image.fromarray(np.uint8(img*255)).save(f'{name}.png')


def detectSoccerField(path, saveImg=False):
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
    Im = EdgeDetection(igs, sigma)
    if saveImg:
        save(Im, "02-edges")
    print("Edges took", round((time.time() - s) * 1000) / 1000)

    print("Hough transform...")
    s = time.time()
    lines = cv.HoughLines(np.uint8(Im*255),rhoRes,thetaRes,int(threshold*255))
    print("CV hough lines took", timeSince(s))
    
        
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
        draw.line(((x1,y1),(x2,y2)),fill="white")
        myCorners = myCorners + findIntersections(
            Line(Point(x1,y1), Point(x2,y2)), i, lines)
    
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
        if (d2 < middle): return first
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

    print("Finding corners took", timeSince(s))
    # Draw rectangle
    draw.line([tr, tl], fill="black")
    draw.line([tl, bl], fill="black")
    draw.line([bl, br], fill="black")
    draw.line([br, tr], fill="black")

    if saveImg:
        Image.fromarray(np.uint8(img)).save(f'04-img-with-hough-lines.png')

    # Todo: find projection matrix

    return colImg, (tl, tr, bl, br), scale


def main():
    # use_open_cv()

    igs, corners, scale = detectSoccerField('static.png', True)
    (topLeft, topRight, bottomLeft, bottomRight) = corners
    img, H = transformImage(igs, corners)


def compute_h(p1, p2):
    x, y = 0, 1
    p = np.empty((len(p1)*2, 9))
    
    for r in range(len(p1)):        
        p[2*r] = np.array([
            [p2[r,x], p2[r,y], 1, 0, 0, 0, -p1[r,x]*p2[r,x], -p1[r,x]*p2[r,y], -p1[r,x]]
        ])
        p[2*r+1] = np.array([
            [0, 0, 0, p2[r,x], p2[r,y], 1, -p1[r,y]*p2[r,x], -p1[r,y]*p2[r,y], -p1[r,y]]
        ])
    
    # With help from https://math.stackexchange.com/a/3511513
    _, _, Vt = np.linalg.svd(p)
    H = Vt[-1].reshape(3, 3)
    
    return H

def compute_h_norm(p1, p2):
    H = compute_h(p1, p2)
    return H

def warp_image(igs_in, igs_ref, H):
    # img in: (row, col) = (y, x) = (h, w)
    in_h, in_w, d = igs_in.shape
    ref_h, ref_w, _ = igs_ref.shape
    
    print("Dimensions", in_h, in_w, ref_h, ref_w)

    # calculate outer bounderies
    warped_edges = (H @ np.array([
        [0, 0, 1],
        [in_w,0,1],
        [0,in_h,1],
        [in_w,in_h,1]
    ]).T)
    warped_edges=warped_edges/warped_edges[-1]
    
    offset_x = -int(min(0, warped_edges[0].min()))
    offset_y = -int(min(0, warped_edges[1].min()))
    mrg_w = int(max(ref_w, warped_edges[0].max())) + offset_x
    mrg_h = int(max(ref_h, warped_edges[1].max())) + offset_y
    
    # init images
    igs_warp = np.zeros((ref_h, ref_w, d))
    igs_merge = np.zeros((mrg_h, mrg_w,d))
    
    # warp into ref and merge result
    for x in range(mrg_w):
        for y in range(mrg_h):
            r = np.linalg.inv(H) @ np.array([x-offset_x, y-offset_y, 1]).T
            xx, yy, _ = r/r[-1]
            i, j = int(xx), int(yy)
            a, b = xx-i, yy-j
            
            if 0 <= i < in_w-1 and 0 <= j < in_h-1:
                c = (1-a)*(1-b) * igs_in[j,i] \
                    + a*(1-b) * igs_in[j,i+1] \
                    + a*b * igs_in[j+1,i+1] \
                    + (1-a)*b * igs_in[j+1,i]
                    
                igs_merge[y, x] = c
                
                if 0 <= x-offset_x < ref_w and 0 <= y-offset_y < ref_h:
                    igs_warp[y-offset_y, x-offset_x] = c
    
    # merge
    igs_merge[offset_y:ref_h+offset_y,offset_x:ref_w+offset_x] = igs_ref[:]
        
    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    H = compute_h_norm(p1, p2)
    igs_rec,_ = warp_image(igs, np.zeros((550,1050,3)), H)
    return igs_rec, H


def transformImage(igs, corners):
   
    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    c_in = np.array([
        [25, 25],
        [1025,25],
        [25, 525],
        [1025,525]
    ])
    c_ref = np.int32(np.array(corners))
    print(c_in, c_ref)
    igs_rec, H = rectify(np.array(igs), c_in, c_ref)

    Image.fromarray(np.uint8(igs_rec)).save(f'05-warped.png')
    
    return igs_rec, H


if __name__ == '__main__':
    main()
