import argparse
import cv2
import os
import time
import openface

fileDir = os.path.dirname(os.path.abspath(__file__))
modelDir = os.path.join(fileDir, '..' ,'openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')

def getRep(frame):
    if frame is None:
        raise Exception("Unable to load image/frame")
    rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bb = align.getAllFaceBoundingBoxes(rgbImg)
    if bb is None:
        return None
    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if len(alignedFaces) == 0:
        return None
    return alignedFaces[0]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=-1,
        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--name', type=str, default='Unknow', required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num', type=int, default=200)

    args = parser.parse_args()
    align = openface.AlignDlib(args.dlibFacePredictor)

    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)
    pathDir = os.path.join(args.output, args.name)
    if not os.path.exists(pathDir):
        os.mkdir(pathDir)
    i = 0
    while True:
        ret, frame = video_capture.read()
        img_align = getRep(frame)
        if img_align is None:
            continue
        path = os.path.join(pathDir, args.name + '_' +str(i)+'.jpg')

        if i > args.num:
            break
        i += 1

        cv2.imwrite(path, frame)

        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.2)

    video_capture.release()
    cv2.destroyAllWindows()