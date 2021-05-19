from ppadb.client import Client as AdbClient
from imutils.object_detection import non_max_suppression
import time
import cv2
import numpy as np
from PIL import Image
import os
import pytesseract
import re
from os import walk

# Equipment list
max_tier = 3
tierPath = []
for i in range(max_tier+1):
    tierPath.append("D:/MEDIA/Code/Python/KyaruAuto/Assets/equipment/Tier"+str(i))
mc = "D:/MEDIA/Code/Python/KyaruAuto/Assets/equipment/useable"
eq = {}
items = []
ij = []
for (dirpath, dirnames, filenames) in walk(mc):
    for j in filenames:
        j = j.replace(".png", "")
        ij.append(j)
    items.extend(ij)

for i in range(max_tier+1):
    ij = []
    for (dirpath, dirnames, filenames) in walk(tierPath[i]):
        for j in filenames:
            j = j.replace(".png", "")
            ij.append(j)
        eq[i] = ij
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

def imageRecognition(image,template,threshold,out,debug = False,multi=False,normalize=False,color=False):
    if color:
        res = cv2.cv2.matchTemplate(image,template,cv2.TM_CCORR_NORMED)
    else: 
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
            x,y = findCenter(startX, endX, startY, endY)
            return y,x
        elif out == 'locr':
            return startY, endY, startX, endX
        elif out == 'bool':
            return max_val > threshold
        else:
            # put error here
            return None

def ocr(image,method,l=None,debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "thresh":
        gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    elif method == "blur":
        gray = cv2.medianBlur(gray, 3)
    elif method == 'adaptive':
        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    filename = "temp/{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    if l == None:
        text = pytesseract.image_to_string(Image.open(filename))
    else:
        text = pytesseract.image_to_string(Image.open(filename),lang=l)
    os.remove(filename)
    if debug:
        return gray
    else :
        return text

def saveSnap(image):
    filename = "screencap/{}.png".format(os.getpid())
    cv2.imwrite(filename, image)
    print('Saved.')
# function
def findCenter(startX,endX,startY,endY):
    # Return x,y
    midh = endX - startX
    midv = endY - startY
    x = startX + (midh/2)
    y = startY + (midv/2)
    return x,y

def tap(x,y):
    cmd = 'input tap {} {}'.format(x,y)
    device.shell(cmd)

def swipe(x,y,x1,y1,d):
    cmd = 'input swipe {} {} {} {} {}'.format(x,y,x1,y1,d)
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

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def buyStamina():
    global stamina_refresh
    tap(788,500)
    stamina_refresh += 1
    time.sleep(1)
    tap(788,500)
    img = screencap()
    while not imageRecognition(img, cv2.imread(r'Assets/ui/stamina-recharged.png',0), 0.8, 'bool'):
        time.sleep(1)
        print('connecting..')
        img = screencap()
    tap(645, 500)
    print('Stamina recharged')
    time.sleep(1)

def shop_buyAll(bonus=True):
    #check if in shop
    img = screencap()
    while not imageRecognition(img, cv2.imread(r'Assets/ui/shop.png',0), 0.9, 'bool'):
        time.sleep(1)
        img = screencap()
        print('Connecting..')
    x = 515
    y = 200
    for i in range(2): # select all
        for j in range(4):
            tap(x,y)
            x += 220
        x = 515
        y += 350
    tap(1050,585) #buy
    time.sleep(1)
    img = screencap()
    i = 0
    while not imageRecognition(img, cv2.imread(r'Assets/ui/shop.png',0), 0.9, 'bool'):
        img = screencap()
        if i < 2:
            while not imageRecognition(img, cv2.imread(r'Assets/ui/ok.png',0), 0.8, 'bool'):
                time.sleep(1)
                print('Connecting...')
                img = screencap()
            y,x = imageRecognition(img, cv2.imread(r'Assets/ui/ok.png',0), 0.8, 'loc')
            tap(x,y)
            i+=1
            time.sleep(1)
    tap(777, 577) #reset
    time.sleep(1)
    tap(787, 500)
    if bonus:
        img = screencap()
        while not imageRecognition(img, cv2.imread(r'Assets/ui/close-bonus.png',0), 0.8, 'bool'):
            time.sleep(1)
            img = screencap()
            print('Connecting...')
        tap(50,50)
    time.sleep(2)
    tap(50, 50)
    print('Done Buy All')

# automation
def stage_menu(enter='dg',item=None):
    global stamina_refresh, stamina_refresh_max, recharge_stamina
    print('entering stage...')
    if enter == 'dg':
        img = screencap()
        while not imageRecognition(img, cv2.imread(r'Assets/ui/start.png',0), 0.8, 'bool'):
            time.sleep(1)
            img = screencap()
        tap(1120, 600)
        return assemble_party()
    elif enter == 'farm':
        img = screencap()
        time.sleep(1)
        item = image_resize(item,height=100)
        img = screencap()
        y,yy,x,xx = imageRecognition(img, cv2.cvtColor(item, cv2.COLOR_BGR2GRAY),0.8 , "locr")
        need = img[y:yy,x:xx]
        need = image_resize(need,height=128)
        tap(1120, 600)
        time.sleep(1)
        img = screencap()
        if imageRecognition(img, cv2.imread(r'Assets/ui/recharge-stamina.png',0), 0.8, 'bool'):
            if recharge_stamina and stamina_refresh_max>stamina_refresh:
                buyStamina()
                time.sleep(1)
                tap(1120, 600)
            else:
                print("No stamina")
                return False
        if assemble_party():
            return findItems(item=need)
        else:
            return False
        #y, x = imageRecognition(img, cv2.cvtColor(need, cv2.COLOR_BGR2GRAY), 0.4, 'loc')
    elif enter == 'grotto':
        img = screencap()
        while not imageRecognition(img, cv2.imread(r'Assets/ui/start.png',0), 0.8, 'bool'):
            time.sleep(1)
            img = screencap()
        tap(1120, 600) 
        assemble_party()
        time.sleep(1)
        tap(888, 655)
        time.sleep(1)
        tap(795,490)
        battle()
        tap(1110, 670)

def assemble_party():
    print('Assembling party...')
    while ocr(screencap()[35:75,508:758], 'thresh')[0:14] != 'Assemble Party':
        time.sleep(1)
    tmplt = cv2.imread(r'Assets/ui/no-chara.png',0)
    if imageRecognition(screencap(), tmplt, 0.997, 'bool',False,True) > 0:
        tap(1140, 120)
        time.sleep(1)
        tap(1040, 230)
    time.sleep(1)
    tap(1115,600)
    print('Entering battle...')
    time.sleep(3)
    return battle()

def win_stage(image):
    tplt_win = cv2.imread(r'Assets/ui/win.png',0)
    tplt_fail = cv2.imread(r'Assets/ui/failed.png',0)
    win = imageRecognition(image, tplt_win, 0.1, 'bool')
    failed = imageRecognition(image, tplt_fail, 0.7, 'bool')
    if failed:
        return False
    elif win:
        return True
    else:
        return None

def battle():
    i = 0
    print('[IN BATTLE]')
    while True:
        time.sleep(1)
        img = screencap() 
        # clock = ocr(img[20:47,1055:1125], 'thresh')[0:5]
        # if clock[0:1] == '1' or clock[0:1] == 'i' or clock[0:1] == '0' or clock[0:1] == 'O' or clock[0:1] == 'o' :
        #     time.sleep(1)
        #     i = 0
        # else :
        # i += 1
        # if i > 3:
        while not imageRecognition(img, cv2.imread(r'Assets/ui/next.png',0), 0.65, 'bool'):
            time.sleep(3)
            img = screencap()
            if imageRecognition(img, cv2.imread(r'Assets/ui/skip.png',0), 0.8, 'bool'):
                print('love-up')
                tap(1195, 50)
            if imageRecognition(img, cv2.imread(r'Assets/ui/level-up.png',0), 0.8, 'bool'):
                print('level-up')
                tap(645, 500) 
            if win_stage(screencap()) == False:
                print('FAILED')
                return False
        print('[END OF BATTLE]')
        tap(1110, 670)
        break
    print('WIN')
    return True
    
def grotto():
    # check if in grotto
    img = screencap()
    while not imageRecognition(img, cv2.imread(r'Assets/ui/grotto.png',0), 0.8, 'bool'):
        time.sleep(1)
        print('Connecting...')
        img = screencap()
    x = 770
    y = 220
    x1 = 1115
    for i in range(2):
        tap(x, y)
        time.sleep(1)
        img = screencap()
        if img[y,x1,0] == 255:
            tap(x1, y)
        else:
            tap(x1, 370)
        time.sleep(1)
        stage_menu(enter='grotto')
        print('done')
    x += 280
    time.sleep(1)
    tap(50,50)
    return

def dungeon(dg=0,enter=False):
    def run(fd):
        img = screencap()
        floor = ocr(img[557:590,275:345], 'thresh')
        if floor[0:2] == '10':
            cur_floor = 10
            max_floor = 10
        elif floor[0:1] == 'S' or floor[0:1] == 's' :
            cur_floor = 5
            max_floor = 10
        else:
            cur_floor = floor[0:1]
            max_floor = floor[2:4]
            cur_floor = int(cur_floor)
            max_floor = int(max_floor)
        while cur_floor <= max_floor:
            img = screencap()
            print(cur_floor,max_floor,sep="/")
            path = "{}floor{}.png".format(folder,cur_floor)
            tmplt = cv2.imread(r"{}".format(path),0)
            y,x = imageRecognition(img, tmplt, 0.8, 'loc')
            tap(x,y-15)
            time.sleep(1)
            result = stage_menu()
            print(result)
            if result:
                time.sleep(5)
                tap(640, 636)
                cur_floor += 1
                if cur_floor > 10:
                    break
                else:
                    print('going next floor')
                    time.sleep(8)
            else :
                print('cancelling...')
                break       
        return
    if enter:
        if dg == 1:
            folder = 'Assets/dungeon/deep_wood/'
            run(folder)
    else:
        img = screencap()
        while not imageRecognition(img, cv2.imread(r'Assets/ui/dungeon.png',0), 0.8, 'bool'):
            time.sleep(1)
            img = screencap()
        if dg == 1:
            folder = 'Assets/dungeon/deep_wood/'
            tap(480, 300)
            time.sleep(1)
            while not imageRecognition(img, cv2.imread(r'Assets/ui/ok.png',0), 0.8, 'bool',True):
                time.sleep(1)
                img = screencap()
            y,x = imageRecognition(img, cv2.imread(r'Assets/ui/ok.png',0), 0.8, 'loc')
            tap(x, y)
            img = screencap()
            while not imageRecognition(img, cv2.imread(r'Assets/ui/deepwood_oak.png',0), 0.8, 'bool'):
                time.sleep(1)
                img = screencap()
            run(folder)
          
def findItems(msc=True,item=None,equipments=True):
    print('evaluating drop...')
    global equipment_get
    found = []
    time.sleep(5)
    img = screencap()
    if imageRecognition(img, cv2.imread(r'Assets/ui/limited-shop.png',0), 0.8, 'bool'):
        print('Limited shop is open')
        global buy_bonus
        if buy_bonus:
            tap(780,500)
            shop_buyAll()
            return True
        else:
            return
    if msc:
        global items
        for i in items:
            if imageRecognition(img, cv2.imread(r'Assets/equipment/useable/{}.png'.format(i),0), 0.9, 'bool') :
                print(i,'Found!')
                if i == 'skip-ticket':
                    global ticket
                    ticket += 1
            img = screencap()
    if equipments:
        global eq
        img = screencap()
        for k in eq:
            print('[{}/{}]Analyzing..'.format(k,len(eq)-1))
            for v in eq[k]:
                tmplt = cv2.imread(r'Assets/equipment/Tier{}/{}.png'.format(k,v))
                if imageRecognition(img, tmplt, 0.97, 'bool',color=True):
                    print(v,"Found")
                    found.append(v)
                    if item is not None:
                        if imageRecognition(item, cv2.cvtColor(tmplt,cv2.COLOR_BGR2GRAY), 0.2, "bool",True):
                            print("Item Found!")
    pattern = r'[0-9]'
    for i in found:
        for j in equipment_get:
            if isnumeric(j[0:1]):
                if isnumeric(name[0:2]):
                    n = int(name[0:2])
                elif isnumeric(name[0:1]):
                    n = int(name[0:1])                
                new_j = re.sub(pattern,'',j)
                new_j = new_j.replace('x', '',1)
                if i == new_j:
                    n += 1
                    equipment_get.append('{}x '+i)

            else:
                equipment_get.append("1x " + i)
                    
 
        # for j in equipment_get:
        #     new_j = re.sub(pattern,'',j)
        #     new_j = new_j.replace('x ', '',1)
        # if i == new_j :
        #     x = equipment_get.index(j)
        #     name = equipment_get[x]

        #     n += 1
        # else:
        #     equipment_get.append("1x "+ i)
    tap(1110,655)
    return True
        
def quest():
    print(stage_menu())
    findItems()

def optimize():
    global run
    run += 1
    print('[{}] Optimizing..'.format(run))
    img = screencap()
    while not imageRecognition(img, cv2.imread(r'Assets/ui/optimize.png',0), 0.7, 'bool'):
        time.sleep(1)
        img = screencap()
    tap(500, 580)
    time.sleep(1)
    img = screencap()
    if imageRecognition(img, cv2.imread(r'Assets/ui/enhance.png',0), 0.7, 'bool'):
        tap(787, 638)
        print('Equipped / leveled up')
        optimize()
    elif imageRecognition(img, cv2.imread(r'Assets/ui/recomended-quest.png',0), 0.8, 'bool'):
        # I need to train my tesseract, so it can read numbers
        #print(ocr(img[330:348,420:450], 'thresh','digits'))
        #debug_show(ocr(img[330:348,420:450], 'thresh',debug=True))
        need = screencap()[265:350,365:445]
        x,y = findCenter(365, 445, 270, 350)
        tap(x,y)
        time.sleep(1)
        if not stage_menu('farm',need):
            return
        optimize()
    elif imageRecognition(img, cv2.imread(r'Assets/ui/refine.png',0), 0.8, 'bool'):
        print('All available equiment already equipped')
        tap(640, 640)
        return

def daily():
    #check menu bar
    img = screencap()
    if imageRecognition(img, cv2.imread(r'Assets/ui/menu-bar.png',0), 0.7, 'bool'):
        if img[700,180,2] == 255:
            print('IN Home')
        else:
            print('Clicking Home')
            tap(180, 700)
            time.sleep(1)
    else :
        print('No menu bar found, please go to home first')
        return
    img = screencap()
    while not imageRecognition(img, cv2.imread(r'Assets/ui/notices.png',0), 0.8, 'bool'):
        time.sleep(1)
        img = screencap()

    # Goto quest
    tap(560,700)
    time.sleep(1)
    # tap grotto
    tap(985,185)
    grotto()
    time.sleep(1)
    tap(1170,180)
    dungeon(dg=1)


# Connect to ADB
client = AdbClient(host="127.0.0.1", port=5037) #connect adb
devices = client.devices() 
if len(devices) == 0: # check if any devices is connected
    print('no device attached')
    quit()

device = devices[0] # sellect first devices
# Initialize
equipment_get = []
stamina_refresh = 1
stamina_refresh_max = 3
recharge_stamina = True
buy_bonus = True
ticket = 0
run = 0
# Runtime

optimize()

# Summary
print("Ticket get : ",ticket)
print("Equipment get : ", equipment_get)
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
