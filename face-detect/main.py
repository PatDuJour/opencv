import cv2 as cv


def clock():
    return cv.getTickCount() / cv.getTickFrequency()


def draw_str(dst, target, s):
    x, y = target
    cv.putText(
        dst,
        s, (x + 1, y + 1),
        cv.FONT_HERSHEY_PLAIN,
        1.0, (0, 0, 0),
        thickness=2,
        lineType=cv.LINE_AA)
    cv.putText(
        dst,
        s, (x, y),
        cv.FONT_HERSHEY_PLAIN,
        1.0, (255, 255, 255),
        lineType=cv.LINE_AA)


def detect(img, cascade):
    rects = cascade.detectMultiScale(
        img,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def main():
    vdcp = cv.VideoCapture(0)

    while True:
        ret, img = vdcp.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        face_cascade = cv.CascadeClassifier(
            './venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml'
        )
        eye_cascade = cv.CascadeClassifier(
            './venv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml'
        )

        t = clock()
        rects = detect(gray, face_cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not eye_cascade.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), eye_cascade)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t
        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv.imshow('facedetect', vis)

    cap.release()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
