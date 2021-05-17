from ppadb.client import Client as AdbClient
from imutils.object_detection import non_max_suppression
import time
import cv2
import numpy as np
from PIL import Image
import os
import pytesseract
# Module
def textDetection(image):
    orig = image.copy()
    (H, W) = image.shape[:2]
    (newW, newH) = (320,320)
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]    
    layerNames = [
	    "feature_fusion/Conv_7/Sigmoid",
	    "feature_fusion/concat_3"]
    net = cv2.dnn.readNet(r"frozen_east_text_detection.pb")
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return orig

def imageRecognition(image,template,threshold,out,debug = False,multi=False,normalize=False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    if normalize:
        cv2.normalize( res, res, 0, 1, cv2.NORM_MINMAX, -1 )
    if multi:
        w, h = template.shape[::-1]
        loc = np.where( res >= threshold)
        i = 0
        for pt in zip(*loc[::-1]):
            i += 1
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
        if out == 'img':
            return image
        if out == 'bool':
            return i
    else:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        (startX, startY) = max_loc
        endX = startX + template.shape[1]
        endY = startY + template.shape[0]
        if debug :
            print(max_val)
        if max_val > threshold:
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
        if out == 'img':
            return image
        elif out == 'loc':
            midh = endX - startX
            midv = endY - startY
            x = startX + (midh/2)
            y = startY + (midv/2)
            return y,x
        elif out == 'bool':
            return max_val > threshold

def ocr(image,method,debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "thresh":
        gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    elif method == "blur":
        gray = cv2.medianBlur(gray, 3)
    filename = "temp/{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    if debug:
        return gray
    else :
        return text

# function
def tap(x,y):
    cmd = 'input tap {} {}'.format(x,y)
    device.shell(cmd)

def screencap():
    capture = device.screencap()
    screen = cv2.imdecode(np.frombuffer(capture, np.uint8),cv2.IMREAD_COLOR)
    return screen

def debug_show(image,name='vision'):
    while True:
        cv2.imshow(name,image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            running = False
            break

# automation
def stage_menu(enter=True):
    if enter:
        tap(1120, 600)
        time.sleep(1)
        return assemble_party()

def assemble_party():
        while ocr(screencap()[35:75,508:758], 'thresh')[0:14] != 'Assemble Party':
            time.sleep(1)
        tmplt = cv2.imread(r'Asset/no-chara.png',0)
        if imageRecognition(screencap(), tmplt, 0.997, 'bool',False,True) > 0:
            tap(1140, 120)
            time.sleep(1)
            tap(1040, 230)
        time.sleep(1)
        tap(1115,600)
        time.sleep(3)
        return battle()

def win_stage(image):
    tplt_win = cv2.imread(r'Asset/win.png',0)
    tplt_fail = cv2.imread(r'Asset/failed.png',0)
    win = imageRecognition(image, tplt_win, 0.1, 'bool')
    failed = imageRecognition(image, tplt_fail, 0.7, 'bool')
    if failed:
        return False
    else:
        return True

def dungeon():
    time.sleep(1)
    img = screencap()
    floor = ocr(img[557:590,275:345], 'thresh')
    if floor[0:2] == '10':
        cur_floor = 10
        max_floor = 10
    else:
        cur_floor = floor[0:1]
        max_floor = floor[2:4]
        cur_floor = int(cur_floor)
        max_floor = int(max_floor)
    while cur_floor <= max_floor:
        print(cur_floor,max_floor,sep="/")
        path = "Asset/dungeon/deep_wood/floor{}.png".format(cur_floor)
        tmplt = cv2.imread(r"{}".format(path),0)
        y,x = imageRecognition(img, tmplt, 0.8, 'loc')
        tap(x,y-10)
        time.sleep(1)
        result = stage_menu()
        print(result)
        if result:
            tap(1110, 670)
            time.sleep(5)
            tap(640, 636)
            cur_floor += 1
            if cur_floor > 10:
                break
            else:
                print('going next floor')
            time.sleep(3)
        else :
            print('cancelling...')
            break
    

def battle():
    i = 0
    while True:
        time.sleep(1)
        img = screencap()
        clock = ocr(img[20:47,1055:1125], 'thresh')[0:5]
        if clock[0:1] == '1' or clock[0:1] == 'i' or clock[0:1] == '0' or clock[0:1] == 'o' :
            print(clock)
            i = 0
        else :
            i += 1
            print('[{}]time not found, retrying...'.format(i))
            if i > 9:
                print('done')
                break
    return win_stage(screencap())
            
           

client = AdbClient(host="127.0.0.1", port=5037) #connect adb
devices = client.devices() 

if len(devices) == 0: # check if any devices is connected
    print('no device attached')
    quit()

device = devices[0] # sellect first devices


dungeon()



""" Main Menu
If active, y = 700
home R = 255, x = 180
Character R = 255, x = 210 
Story G = 239, x = 380
Quest R = 165, x = 560
Guildhouse R = 255, x = 760
Gacha R = 247, x = 940
Menu R = 165, x = 1110

cords y = 580
shop x = 820
clan x = 920
mission x = 1115 
present x = 1212
buy stamina y = 40 x = 430 """

# tplt = cv2.imread(r"Asset/equipment/moonlight-sword-blu.png",0)

# while True:
#     # img = cv2.imread(r"screencap/reward-get-obsc.jpg")
#     result = device.screencap()
#     img = cv2.imdecode(np.frombuffer(result, np.uint8),cv2.IMREAD_COLOR)
#     res = imageRecognition(img, tplt,0.1,'img',True)

#     # txt = ocr(img, 'thresh')
#     # print(txt)

#     cv2.imshow("vision",res)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         running = False
#         break





