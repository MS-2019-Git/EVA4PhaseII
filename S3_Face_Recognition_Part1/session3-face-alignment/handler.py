try:
    import unzip_requirements
except ImportError:
    pass
	
import cv2
import dlib
import numpy as np
# from PIL import Image
import math
import boto3
import os
import io
import json
import base64
from requests_toolbelt.multipart import decoder
print('Import END...')


# s3 = boto3.client('s3')
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

datname = r'shape_predictor_5_face_landmarks.dat'
def dlibLandmarksToPoints(shape):
  points = []
  for p in shape.parts():
    pt = (p.x, p.y)
    points.append(pt)
  return points

def getLandmarks(faceDetector, landmarkDetector, im, FACE_DOWNSAMPLE_RATIO = 1):
  points = []
  imSmall = cv2.resize(im,None,
                       fx=1.0/FACE_DOWNSAMPLE_RATIO,
                       fy=1.0/FACE_DOWNSAMPLE_RATIO,
                       interpolation = cv2.INTER_LINEAR)
  faceRects = faceDetector(imSmall, 0)
  if len(faceRects) > 0:
    maxArea = 0
    maxRect = None
    # TODO: test on images with multiple faces
    for face in faceRects:
      if face.area() > maxArea:
        maxArea = face.area()
        maxRect = [face.left(),
                   face.top(),
                   face.right(),
                   face.bottom()
                  ]
    rect = dlib.rectangle(*maxRect)
    scaledRect = dlib.rectangle(int(rect.left()*FACE_DOWNSAMPLE_RATIO),
                             int(rect.top()*FACE_DOWNSAMPLE_RATIO),
                             int(rect.right()*FACE_DOWNSAMPLE_RATIO),
                             int(rect.bottom()*FACE_DOWNSAMPLE_RATIO))
    landmarks = landmarkDetector(im, scaledRect)
    points = dlibLandmarksToPoints(landmarks)
  return points

def similarityTransform(inPoints, outPoints):
  s60 = math.sin(60*math.pi/180)
  c60 = math.cos(60*math.pi/180)
  inPts = np.copy(inPoints).tolist()
  outPts = np.copy(outPoints).tolist()
  # The third point is calculated so that the three points make an equilateral triangle
  xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
  yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
  inPts.append([np.int(xin), np.int(yin)])
  xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
  yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
  outPts.append([np.int(xout), np.int(yout)])
  tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
  return tform

def normalizeImagesAndLandmarks(outSize, imIn, pointsIn):
  h, w = outSize
  if len(pointsIn)==36:
    eyecornerSrc = [pointsIn[36], pointsIn[45]]
  else:
    eyecornerSrc = [pointsIn[2], pointsIn[0]]
  eyecornerDst = [(np.int(0.3 * w), np.int(h/3)),
                  (np.int(0.7 * w), np.int(h/3))]
  tform = similarityTransform(eyecornerSrc, eyecornerDst)
  imOut = np.zeros(imIn.shape, dtype=imIn.dtype)
  imOut = cv2.warpAffine(imIn,tform, (w, h))
  points2 = np.reshape(pointsIn, (pointsIn.shape[0], 1, pointsIn.shape[1]))
  pointsOut = cv2.transform(points2, tform)
  pointsOut = np.reshape(pointsOut, (pointsIn.shape[0], pointsIn.shape[1]))
  return imOut, pointsOut

def get_prediction(image_bytes):
    faceDetector=dlib.get_frontal_face_detector()
    landmarkDetector=dlib.shape_predictor(datname)
    print('facedetector and landmarkdetector object done')
    img = np.frombuffer(image_bytes, dtype=np.int8)
    # img = np.asarray(image_bytes, dtype='uint8')
    # img = Image.open(io.BytesIO(image_bytes))
    print('img converted to array done')
    im = cv2.imdecode(img, cv2.IMREAD_COLOR)
    # print(im)
    points = getLandmarks(faceDetector,landmarkDetector,im)
    points=np.array(points)
    im = np.float32(im)/255
    h=600;w=600;
    print(points)
    imNorm,points=normalizeImagesAndLandmarks((h,w),im,points)
    imNorm = np.uint8(imNorm*255)
    print("imNorm shape")
    print(imNorm.shape)
    data_serial = cv2.imencode('.png', imNorm)[1].tostring()
    print(data_serial)
    s3.Object('session3-face-alignment-face-swap-mou', 'test1.png').put(Body=data_serial,ContentType='image/PNG', ACL='public-read')
    url = "https://{}.s3.amazonaws.com/{}".format('session3-face-alignment-face-swap-mou', 'test1.png')
    # s3.Bucket('my-pocket').put_object(Key='cluster.png',Body=out_img,ContentType='image/png',ACL='public-read')
    print(url)
    return url

def align_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        #print(event['body'])
        body = base64.b64decode(event["body"])
        print('BODY LOADED')
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        print("Picture loaded")	
        prediction = get_prediction(image_bytes=picture.content)
        print("Prediction done")
        print(prediction)
        
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename)<4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type':'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predicted class id':prediction})
        }

    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type':'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials':True
            },
            "body": json.dumps({"error": repr(e)})
        }
