import heapq
import math
import time

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
from sympy import Line, Point

sigma = 1.6
threshold = 0.2
rhoRes = 2
thetaRes = math.pi/180
nLines = 18
resizeWidth = 840


def BorderExtendPad(Igs, k):
    outImg = np.zeros((len(Igs) + 2*k, len(Igs[0]) + 2*k))

    outImg[k:-k, k:-k] = Igs

    for i in range(k):
        for j in range(k):
            outImg[i][j] = Igs[0][0]
            outImg[-i-1][j] = Igs[-1][0]
            outImg[i][-j-1] = Igs[0][-1]
            outImg[-i-1][-j-1] = Igs[-1][-1]

        for j in range(len(Igs)):
            outImg[k+j][i] = Igs[j][0]
            outImg[k+j][-i-1] = Igs[j][-1]

        for j in range(len(Igs[0])):
            outImg[i][k+j] = Igs[0][j]
            outImg[-i-1][k+j] = Igs[-1][j]

    return outImg


def GaussianKernel(k, sigma):
    kernel = np.zeros((2*k + 1, 2*k + 1))
    sigSqeDbl = (sigma**2) * 2

    def g(i, j):
        return math.exp(- (i*i + j*j) / sigSqeDbl
                        ) / (sigSqeDbl * math.pi)

    for i in range(-k, k+1):
        for j in range(-k, k+1):
            kernel[i+k, j+k] = g(i, j)

    return kernel


def ConvFilter(Igs, G):
    k = (len(G) - 1) // 2
    IgsPad = BorderExtendPad(Igs, k)
    Iconv = np.zeros((len(Igs), len(Igs[0])))
    krn_rng = range(-k, k+1)

    for i in range(len(Igs)):
        for j in range(len(Igs[0])):
            Iconv[i, j] = sum([
                G[m+k, n+k] * IgsPad[i - (m-k), j - (n-k)]
                for m in krn_rng
                for n in krn_rng
            ])

    return Iconv


def SobelKernel(d="h"):
    if d == "h":
        return np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])
    elif d == "v":
        return np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
    else:
        raise "Invalid argument!"


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


def EdgeDetection(Igs, sigma):
    h, w = len(Igs), len(Igs[0])
    
    # Todo: fast gaussian / blur
    print("  Gaussian...")
    imgSmooth = ConvFilter(Igs, GaussianKernel(2, sigma))

    # print("  Sobel...")
    # Ix = ConvFilter(imgSmooth, SobelKernel("h"))
    # Iy = ConvFilter(imgSmooth, SobelKernel("v"))

    print("  Sobel with cv...")
    s = time.time()
    sobelx = cv.Sobel(Igs, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(Igs, cv.CV_64F, 0, 1, ksize=3)
    print("  Took", time.time() - s)

    # Why do I have to switch?
    Ix, Iy = sobely, sobelx

    Io = np.zeros((h, w))
    Im = np.zeros((h, w))
    Ims = np.zeros((h, w))

    print("  Magnitude and direction...")
    for i in range(h):
        for j in range(w):
            Im[i, j] = math.sqrt(Ix[i, j]**2 + Iy[i, j]**2)
            Io[i, j] = np.arctan2(Ix[i, j], Iy[i, j]) - math.pi / 2

    print("  Non-max suppression...")
    for i in range(h):
        for j in range(w):
            Ims[i, j] = SuppressNonMax(Im, i, j, Io[i, j])

    Ims /= Ims.max()
    return Ims, Io, Ix, Iy


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


def findIntersections(line, k, rho, theta):
    found = []
    for i in range(len(rho)):
        if i != k:
            corners = getLine(rho[i], theta[i]).intersection(line)
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
    img = Image.open(path).convert('L')

    # Resize
    w, h = img.size
    scale = resizeWidth / w
    img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

    # Normalize and suppress dark regions
    igs = np.array(img) / 255.
    igs = np.maximum(igs, 0.5)

    if saveImg:
        save(igs, "01-grayscale")

    print("Detect edges...")
    Im, Io, Ix, Iy = EdgeDetection(igs, sigma)
    if saveImg:
        save(Im, "02-edges")

    print("Hough transform...")
    s = time.time()
    H = HoughTransform(Im, threshold, rhoRes, thetaRes)
    print("Took ", time.time() - s)
    if saveImg:
        save(H, "03-hough-transform-map")

    print("Hough lines...")
    lRho, lTheta = HoughLines(H, rhoRes, thetaRes, nLines)
    print(len(lRho), " lines found.")

    draw = ImageDraw.Draw(img)

    # Todo: only compare with promising candidates
    # (ie rule out lines with same orientation or in the middle of the field)
    print("Find intersections...")
    myCorners = []
    for k in range(len(lRho)):
        r, t = lRho[k], lTheta[k]
        mag, rot = r, t - math.pi/2

        drawLine(draw, r, t - math.pi/2, "black")

        p1 = Point(+mag*math.cos(rot) - 1000*math.sin(rot), -mag *
                   math.sin(rot) - 1000*math.cos(rot), evaluate=False)
        p2 = Point(+mag*math.cos(rot) + 1000*math.sin(rot), -mag *
                   math.sin(rot) + 1000*math.cos(rot), evaluate=False)
        ln = Line(p1, p2)

        myCorners = myCorners + findIntersections(ln, k, lRho, lTheta)

    tl = (-1, -1)
    tr = (-1, -1)
    bl = (-1, -1)
    br = (-1, -1)

    for i in range(len(myCorners)):
        x, y = myCorners[i]
        if 0 < x < resizeWidth/2 and 0 < y < h*resizeWidth/w/2:
            tl = (x, y)
        elif resizeWidth/2 < x < resizeWidth and 0 < y < h*resizeWidth/w/2:
            tr = (x, y)
        elif 0 < x < resizeWidth/2 and h*resizeWidth/w/2 < y < h*resizeWidth/w:
            bl = (x, y)
        elif resizeWidth/2 < x < resizeWidth and h*resizeWidth/w/2 < y < h*resizeWidth/w:
            br = (x, y)

    # Draw rectangle
    draw.line([tr, tl], fill="white")
    draw.line([tl, bl], fill="white")
    draw.line([bl, br], fill="white")
    draw.line([br, tr], fill="white")

    if saveImg:
        Image.fromarray(np.uint8(img)).save(f'04-img-with-hough-lines.png')

    # Todo: find projection matrix

    return (tl, tr, bl, br), scale


def main():
    # use_open_cv()

    corners, scale = detectSoccerField('static.png', True)
    (topLeft, topRight, bottomLeft, bottomRight) = corners


def use_open_cv():
    img = cv.imread("static.png", cv.IMREAD_COLOR)
    img = cv.blur(img, (11, 11))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imwrite('houghlines3.jpg', img)


if __name__ == '__main__':
    main()
