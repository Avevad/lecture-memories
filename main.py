import os

import cv2
import numpy as np
import matplotlib.pyplot as mpl
import sys


def usage():
    print(f"Usage: {sys.argv[0]} <FILE> <SHOTS_COUNT> <INVERT>")


if len(sys.argv) != 4 or not sys.argv[2].isnumeric():
    usage()
    exit(-1)

FILE_NAME = sys.argv[1]
SELECTION_SIZE = int(sys.argv[2])
INVERT = sys.argv[3].lower() == "true"
OUTPUT_FOLDER = "output/"
OUTPUT_PREFIX = "frame"
OUTPUT_POSTFIX = ".png"
DEBUG = len(sys.argv) > 4
FW, FH = 640, 480
CROP = 0.1
SMOOTHING = 20
SMOOTHING_ITERS = 5
SKIP = 239
OFF = 2
BOARD_THRESHOLD = 170
NOTES_THRESHOLD = 150
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe("model.prototxt", "model.caffemodel")


def get_person_bounds(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if CLASSES[idx] == "person":
                return startX, startY, endX, endY
    return 0, 0, 0, 0


def get_blackboard_bounds(frame_grayscale):
    if DEBUG:
        cv2.imshow("frame_grayscale", frame_grayscale)
    contours, hierarchy = cv2.findContours(frame_grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0
    for i, contour in enumerate(contours):
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw * ch > w * h:
            x, y, w, h = cx, cy, cw, ch
    return x, y, w, h


def remove_person(frame, px1, py1, px2, py2):
    removed = frame.copy()

    for y in range(py1 + 1, py2):
        cv2.line(removed, (px1, y), ((px1 + px2) // 2, y), tuple(int(c) for c in frame[y, px1]), 1)
        cv2.line(removed, ((px1 + px2) // 2, y), (px2, y), tuple(int(c) for c in frame[y, px2]), 1)

    return removed


def get_blackboard_completeness(removed, px1, py1, px2, py2, bx, by, bw, bh):
    gray = cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY)
    ret, baw_notes = cv2.threshold(gray, NOTES_THRESHOLD, 255, cv2.THRESH_BINARY if INVERT else cv2.THRESH_BINARY_INV)

    cv2.rectangle(baw_notes, (px1, py1), (px2, py2), (255, 255, 255), cv2.FILLED)

    baw_notes = baw_notes[by:by + bh, bx:bx + bw]

    if DEBUG:
        cv2.imshow("blackboard", baw_notes)

    completeness = 1 - cv2.countNonZero(baw_notes) / (bw * bh)

    return completeness


def get_frame_params(frame):
    px1, py1, px2, py2 = get_person_bounds(frame)
    px1, py1, px2, py2 = min(px1 - OFF, FW - 1), min(py1 - OFF, FH - 1), min(px2 + OFF, FW - 1), min(py2 + OFF, FH - 1)

    removed = remove_person(frame, px1, py1, px2, py2)

    gray = cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY)
    ret, baw_board = cv2.threshold(gray, BOARD_THRESHOLD, 255, cv2.THRESH_BINARY if INVERT else cv2.THRESH_BINARY_INV)
    bx, by, bw, bh = get_blackboard_bounds(baw_board)
    bxc = by + int(CROP * bw)
    byc = bx + int(CROP * bh)
    bwc = bw - int(CROP * bw * 2)
    bhc = bh - int(CROP * bh * 2)

    c = get_blackboard_completeness(removed, px1, py1, px2, py2, bxc, byc, bwc, bhc)

    return px1, py1, px2, py2, bx, by, bx + bw, by + bh, c


def process_video(video):
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if DEBUG:
        mpl.ion()
        mpl.show()

    ppx1, ppy1, ppx2, ppy2 = 0, 0, 0, 0
    xs = []
    ys = []
    bs = []
    f = 0
    while True:
        print("%.2f%%" % (f / frames * 100))
        video.set(cv2.CAP_PROP_POS_FRAMES, f + SKIP)
        ret, frame = video.read()
        f += SKIP + 1
        if not ret:
            break
        (h, w) = frame.shape[:2]
        small = cv2.resize(frame, (640, 480))
        px1, py1, px2, py2, bx1, by1, bx2, by2, c = get_frame_params(small)
        px1 *= w / FW
        py1 *= h / FH
        px2 *= w / FW
        py2 *= h / FH
        bx1 *= w / FW
        by1 *= h / FH
        bx2 *= w / FW
        by2 *= h / FH
        bs.append((int(bx1), int(by1), int(bx2), int(by2)))
        if px1 == 0 and py1 == 0 and px2 == 0 and py2 == 0:
            px1, py1, px2, py2 = ppx1, ppy1, ppx2, ppy2
        px1, py1, px2, py2, bx1, by1, bx2, by2 = \
            int(px1), int(py1), int(px2), int(py2), int(bx1), int(by1), int(bx2), int(by2)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 5)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 5)
        if DEBUG:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        ppx1, ppy1, ppx2, ppy2 = px1, py1, px2, py2
        xs.append(len(xs))
        ys.append(c)
        if DEBUG:
            mpl.plot(xs, ys, color="blue")
            mpl.draw()
            mpl.pause(0.001)
    return ys, bs


def smooth(cs):
    w = 0
    s = 0
    r = []
    for i in range(len(cs)):
        if w < SMOOTHING:
            w += 1
        else:
            s -= cs[i - w]
        s += cs[i]
        r.append(s / w)
    return r


track = cv2.VideoCapture(FILE_NAME)
if not track.isOpened():
    print(f"error: file {FILE_NAME} not found")
    exit(-1)
comps, boards = process_video(track)
smoothed = comps
for s in range(SMOOTHING_ITERS):
    smoothed = smooth(smoothed)

if DEBUG:
    mpl.ioff()
    mpl.close()
    mpl.plot(smoothed)
    mpl.show()

comps = smoothed
comps = enumerate(comps)
ncomps = []
raising = True
prev = 0
for i, x in comps:
    if raising:
        if x < prev:
            ncomps.append((i, x))
            raising = False
    else:
        if x > prev:
            raising = True
    prev = x
comps = ncomps
comps = sorted(comps, key=lambda t: t[1])
comps = list(reversed(comps))
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)
for i in range(min(SELECTION_SIZE, len(comps))):
    j, comp = comps[i]
    track.set(cv2.CAP_PROP_POS_FRAMES, (SKIP + 1) * j + SKIP)
    ret, cframe = track.read()
    cframe = cframe[boards[j][1]:boards[j][3], boards[j][0]:boards[j][2]]
    if ret:
        cv2.imwrite(f"{OUTPUT_FOLDER}{OUTPUT_PREFIX}{i}{OUTPUT_POSTFIX}", cframe)
