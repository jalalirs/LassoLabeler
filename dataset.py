
import json
import os
import random
import numpy as np
from PIL import Image
import cv2
VALID_FORMAT = ('.BMP', '.GIF', '.JPG', '.JPEG', '.PNG', '.PBM', '.PGM', '.PPM', '.TIFF', '.XBM')  # Image formats supported by Qt
VALID_VIDEO_FORMAT = (".MP4",".MOV")
import matplotlib.pyplot as plt
from utils import notify
import imageio

# starting from 1 to eliminate any chance of having 0,0,0

create_random_color = lambda : (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
COLORS = [create_random_color() for i in range(1000)]
get_color = lambda o: COLORS[o]


def exists(path):
    return os.path.exists(path)

class Video:
    def __init__(self,path):
        self._path = path
        self._counter = -1
        self._numOfFrames = -1
        self._fps = -1
        self._open = False
        self._cap = None
    
    def open(self):
        self._cap = cv2.VideoCapture(self._path)
        if self._cap.isOpened() == False:
            notify("Error opening video stream or file","error")
            self._open = False
            return False

        self._numOfFrames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self._counter = 0
        self._open = True

        return True
    
    def read(self):
        self._cap.set(1,self._counter)
        ret, frame = self._cap.read()

        if not ret:
            return False, None

        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGRA2RGB)
        return ret, frame
    
    def goto(self,frameNum):
        if frameNum >= self._numOfFrames and frameNum < 0:
            notify(f"Frame number should be greater than 0 and less than {self._numOfFrames}","error")
            return False,None 
        self._counter = frameNum
        return self.read()

    def readNext(self,jump=None):
        if jump is None:
            jump = 1
        
        if self._counter + jump < self._numOfFrames:
            self._counter += jump
            return self.read()
        else:
            notify(f"No more frames with current jump {jump}","error")
            return False,None   

    def readPrev(self,jump=None):
        if jump is None:
            jump = 1
        
        if self._counter - jump >= 0:
            self._counter -= jump
            return self.read()
        else:
            notify(f"No more frames with current jump {jump}","error")
            return False,None

    def close(self):
        self._cap.release() 
        self._counter = -1
        self._numOfFrames = -1
        self._fps = -1
        self._open = False
        self._cap = None
        return True
    
    def isOpen(self):
        return self._open
    
    def numOfFrames(self):
        return self._numOfFrames
    
    def currentFrame(self):
        return self._counter

    @classmethod
    def create(self,path):
        return Video(path)

class DatasetObject:
    def __init__(self,name,color):
        self._name = name 
        self._color = color

    def color(self):
        return self._color

    def __str__(self):
        return self._name, self._color

class Annotation:
    def __init__(self,path):
        self._path = path
        self._shapes = {}
        self._objectShapes = {}
        self._objects = {}
        self._shapeCounter = 0 # act as counter for shapes

    def addShape(self,label,shapeStr,points,objectId):
        
        if shapeStr == "rectangle":
            points = [[min(points[0][0],points[1][0]),min(points[0][1],points[1][1])],
                        [max(points[0][0],points[1][0]),max(points[0][1],points[1][1])]],    
        shape = {
                "label": label,
                "points": points,
                "group_id": objectId,
                "shape_type":shapeStr,
                "flags": {}
        }

        self._shapes[self._shapeCounter] = shape
        if objectId in self._objectShapes:
            self._objectShapes[objectId].append(self._shapeCounter)
        else:
            self._objectShapes[objectId] = [self._shapeCounter]
        
        
        # will not be written or used. Just to aid in temprorly created objects without shapes
        # mainly used by objectNames()
        if objectId not in self._objects:
            self._objects[objectId] = DatasetObject(objectId,get_color(len(self._objects)))
        
        self._shapeCounter += 1
        
    def deleteShape(self,objectId,contourId):
        index = self._objectShapes[objectId][contourId]
        self._objectShapes[objectId].pop(contourId)
        self._shapes.pop(index)      

    def shapes(self):
        return list(self._shapes.values())
    
    def getObjectShapes(self,oId=None,shapeType=None):
        if oId is None:
            return self._objectShapes

        if oId in self._objectShapes:
            if shapeType is None:
                return self._objectShapes[oId]
            else:
                return [k for k in self._objectShapes[oId] if self._shapes[k]["shape_type"] == shapeType]

        return []
    
    def getShape(self,index):
        return self._shapes[index]

    def save(self,imgPath,width,height,boundingBox=False):
        ann = {
            "version": "4.5.6",
            "flags": {},
            "shapes": list(self._shapes.values()),
            "imagePath": imgPath,
            "imageData":None,
            "imageHeight": height,
            "imageWidth": width
        }
        if boundingBox:
            for o,s in self._objectShapes.items():
                if len(s) == 0:
                    continue
                s1 = self._shapes[s[0]]
                points = s1["points"]
                polygon = np.array(points)
                x1,y1,x2,y2 = polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()
                for s2 in s[1:]:
                    points = self._shapes[s2]["points"]
                    polygon = np.array(points)
                    x1,y1 = min(x1,polygon[:,0].min()),min(y1,polygon[:,1].min())
                    x2,y2 = max(x2,polygon[:,0].max()),max(y2,polygon[:,1].max())
                bs = {
                    "label": s1["label"],
                    "points": [[x1,y1],[x2,y2]],
                    "group_id": s1["group_id"],
                    "shape_type":"rectangle",
                    "flags": {}
                }
                ann["shapes"].append(bs)
        
        with open(self._path,"w") as f:
            f.write(json.dumps(ann,indent=4))

    def getColor(self,objectId):
        return self._objects[objectId].color()

    def getObjectNames(self):
        return list(self._objects.keys())

    @classmethod
    def fromJson(self,path):
        if exists(path):
            with open(path) as f:
                annotationDict = json.load(f)
            annotation = Annotation(path)
            for s in annotationDict["shapes"]:
                if s["shape_type"] == "polygon":
                    annotation.addShape(s["label"],s["shape_type"],s["points"],s["group_id"])     
        else:
           annotation = Annotation(path) 
        return annotation

class DatasetItem:
    def __init__(self,name,imgPath,annotationPath,maskPath,itemid):
        self._name = name
        self._id = itemid
        self._imgPath = imgPath
        self._annotationPath = annotationPath
        self._maskPath = maskPath
        self._img = None
        self._imgArray = None
        self._viewArray = None
        self._annotation = None
        self._maskColor = None
        self._contourFilling = None
        self._imgbase64 = ""
        self._labelsCount = {}
        self._changed = False
        
    def drawContourOnMask(self,points,color):
        polygon = np.array(points)
        polygon = polygon.reshape((-1,1,2)).astype(np.int32)
        self._maskColor = cv2.drawContours(self._maskColor, [polygon], -1, color=color, thickness=5)
    
    def updateMask(self):
       self._maskColor = np.zeros_like(self._maskColor)
       for s in self.annotation().shapes():
            objectId = s["group_id"]
            points = s["points"]
            color = self.annotation().getColor(objectId)
            self.drawContourOnMask(points,color)

    def addShape(self,label,shapeStr,points,objectId):
        self.annotation().addShape(label,shapeStr,points,objectId)
        color = self.annotation().getColor(objectId)
        self.drawContourOnMask(points,color)
        self._changed = True
    
    def image(self):
        image = self._imgArray.copy() 
        borderMask = np.where(self._maskColor != [0,0,0])
        fillingMask = np.where(self._contourFilling != [0,0,0])
        image[borderMask] = self._maskColor[borderMask]
        if fillingMask[0].shape[0] > 0:
            filling = (image[fillingMask].astype(float) + 0.4*self._contourFilling[fillingMask].astype(float)).clip(0,255).astype(np.uint8)
            image[fillingMask] = filling
        return image
    
    def mask(self):
        return self._maskArray
    
    def maskImage(self):
        img = np.zeros_like(self._imgArray)
        objectShapes = self.annotation().getObjectShapes()
        for o,shapeIdxs in objectShapes.items():
            color = self.annotation().getColor(o)
            for index in shapeIdxs:
                shape = self.annotation().getShape(index)
                points = shape["points"]
                polygon = np.array(points)
                polygon = polygon.reshape((-1,1,2)).astype(np.int32)
                img = cv2.drawContours(img, [polygon], -1, color=color, thickness=cv2.FILLED)
        return img
    
    def boundingboxImage(self):
        img = self._imgArray.copy()
        objectShapes = self.annotation().getObjectShapes()
        for o,shapeIdx in objectShapes.items():
            color = self.annotation().getColor(o)
            if len(shapeIdx) == 0:
                continue
            s1 = self.annotation().getShape(shapeIdx[0])
            points = s1["points"]
            polygon = np.array(points)
            x1,y1,x2,y2 = polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()
            for s2idx in shapeIdx[1:]:
                s2 = self.annotation().getShape(s2idx)
                points = s2["points"]
                polygon = np.array(points)
                x1,y1 = min(x1,polygon[:,0].min()),min(y1,polygon[:,1].min())
                x2,y2 = max(x2,polygon[:,0].max()),max(y2,polygon[:,1].max())
            
            img = cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color=color, thickness=2)         
        return img

    def annotation(self):
        return self._annotation

    def open(self):
        self._img = Image.open(self._imgPath)
        self._imgArray = np.asarray(self._img)
        w,h, c = self._imgArray.shape
        self._maskColor = np.zeros_like(self._imgArray)
        self._contourFilling = np.zeros_like(self._imgArray)
        self._labelsCount = {}

        self._annotation = Annotation.fromJson(self._annotationPath)
        for s in self.annotation().shapes():
            objectId = s["group_id"]
            points = s["points"]
            color = self.annotation().getColor(objectId)
            self.drawContourOnMask(points,color)
        
        for o in self.annotation().getObjectNames():
            label = "_".join(objectId.split("_")[:-1])
            count = int(objectId.split("_")[-1])
            if label in self._labelsCount:
                nameCount, instanceCount = self._labelsCount[label]
                self._labelsCount[label] = (max(nameCount,count),instanceCount+1)
            else:
                self._labelsCount[label] = (count,1)
    
    def close(self):
        self._img.close()
        self._img = None
        self._annotation = None
        self._mask = None
        self._imgArray = None
        self._changed = False
    
    def save(self,boundingBox=False):
        height,width,_ = self._imgArray.shape
        imgRelativePath = f"../imgs/{self._imgPath.split('/')[-1]}"
        self._annotation.save(imgRelativePath,width,height,boundingBox)

        #self._mask.save()
        
    def mask(self,refresh=False):
        if self._mask and not refresh:
            return self._mask
        
        shapes = self.annotation().shapes()
        nObjects = self.annotation().objectsCount()
        mask = np.zeros_like(self.image())
        for o in range(nObjects):
            color = get_color(o)
            oShapes = self.annotation().getObjectShapes(o,"polygon")
            for s in oShapes:
                polygon = np.array(s["points"])
                polygon = polygon.reshape((-1,1,2)).astype(np.int32)
                mask= cv2.drawContours(mask, [polygon], -1, color=color, thickness=cv2.FILLED)
        return mask         

    def objectNames(self):
        return self.annotation().getObjectNames()
    
    def shapesForObject(self,objectId,shapeType=None):
        objectShapes = self.annotation().getObjectShapes(objectId,shapeType)
        return len(objectShapes)

    def fillInContour(self,objectId,contourIndex):
        index = self.annotation().getObjectShapes(objectId,"polygon")[contourIndex]
        points = self.annotation().getShape(index)["points"]
        
        # single filling only
        self._contourFilling = np.zeros_like(self._contourFilling)
        polygon = np.array(points)
        polygon = polygon.reshape((-1,1,2)).astype(np.int32)
        self._contourFilling = cv2.drawContours(self._contourFilling, [polygon], -1, color=(0,0,255), thickness=cv2.FILLED)
    
    def getContourBoundingBox(self,objectId,contourIndex):
        index = self.annotation().getObjectShapes(objectId,"polygon")[contourIndex]
        points = self.annotation().getShape(index)["points"]
        polygon = np.array(points)
        x1,y1,x2,y2 = polygon[:,0].min(),polygon[:,1].min(),polygon[:,0].max(),polygon[:,1].max()
        return int(x1),int(y1),int(x2),int(y2)

    def deleteContour(self,objectId,contourIndex):
        self.annotation().deleteShape(objectId,contourIndex)
        self.updateMask()
        self._contourFilling = np.zeros_like(self._contourFilling)
        self._changed = True

    def createObject(self,label):
        if label in self._labelsCount:
            nameCount, instanceCount = self._labelsCount[label]
            self._labelsCount[label] = nameCount+1, instanceCount+1
        else:
            self._labelsCount[label] = 1,1
        self._changed = True
        return f'{label}_{self._labelsCount[label][0]}'

    def didChange(self):
        return self._changed
    
    def id(self):
        return self._id

    @classmethod
    def create(cls,datasetPath,name,fileName,itemid):
        imgPath = f"{datasetPath}/imgs/{fileName}"
        annotationPath = f"{datasetPath}/annotations/{name}.json"
        maskPath = f"{datasetPath}/mask/{name}.jpg"
        return DatasetItem(name,imgPath,annotationPath,maskPath,itemid)

class Key:
    def __init__(self,name,imagePath):
        self._name = name
        self._imagePath = imagePath
        self._count = 0
    
    def incr(self):
        self._count += 1

    def decr(self):
        self._count -= 1

    def count(self):
        return self._count

    def image(self):
        return self._imagePath

    @classmethod
    def create(self,datasetPath,name,keyFileName):
        return Key(name,f'{datasetPath}/keys/{keyFileName}')

class Dataset:
    def __init__(self,path,imgFiles,keyFiles,videoFiles=None):
        self._path = path
        self._itemNames = [im.split(".")[0] for im in imgFiles]
        self._keysName = [key.split(".")[0] for key in keyFiles]
        self._items = {name:DatasetItem.create(path,name,img,imgid) for name,img,imgid in zip(self._itemNames,imgFiles,range(len(imgFiles)))}
        self._keys = {name:Key.create(path,name,key) for name,key in zip(self._keysName,keyFiles)}
        self._currentItem = None
        if videoFiles is not None:
            self._videoNames = [vi.split(".")[0] for vi in videoFiles]
            self._videos = {name: Video.create(f"{path}/videos/{vfile}") for name,vfile in zip(self._videoNames,videoFiles)}
        else:
            self._videoNames = []
            self._videos = {}
        
    def names(self):
        return self._names
    
    def changeItem(self,newName,save=True):
        if save and self._currentItem:
            self._currentItem.save()
        if self._currentItem:
            self._currentItem.close()
        self._currentItem = self._items[newName]
        self._currentItem.open()
    
    def currentImage(self):
        return self._currentItem.image()
    
    def currentMaskImage(self):
        return self._currentItem.maskImage()

    def currentBoundingboxImage(self):
        return self._currentItem.boundingboxImage()

    def currentMask(self):
        return self._currentItem.mask()
    
    def currentAnnotation(self):
        return self._currentItem.annotation()
    
    def addShape(self,label,shapeStr,points,objectId):
        self._currentItem.addShape(label,shapeStr,points,objectId)

    def keys(self):
        return list(self._keys.keys())
    
    def keyCount(self,name):
        return self._keys[name].count()
    
    def keyImage(self,name):
        return self._keys[name].image()
    
    def keyIncr(self,name):
        self._keys[name].incr()

    def keyDecr(self,name):
        self._keys[name].decr()

    def itemNames(self):
        return list(self._items.keys())

    def objectNames(self):
        return self._currentItem.objectNames()
    
    def videos(self):
        return self._videoNames

    def shapesForObject(self,objectId,shapeType=None):
        return self._currentItem.shapesForObject(objectId,shapeType)
    
    def save(self,boundingBox=False):
        self._currentItem.save(boundingBox)
    
    def fillInContour(self,currentObject,contourId):
        self._currentItem.fillInContour(currentObject,contourId)

    def getContourBoundingBox(self,currentObject,contourId):
        return self._currentItem.getContourBoundingBox(currentObject,contourId)
    
    def deleteContour(self,objectId,contourIndex):
        self._currentItem.deleteContour(objectId,contourIndex)
    
    def createObject(self,label):
        return self._currentItem.createObject(label)
    
    def didChange(self):
        return self._currentItem.didChange()

    def openVideo(self,videoId):
        return self._videos[videoId].open()
    
    def closeVideo(self,videoId):
        return self._videos[videoId].close()
    
    def isVideoOpen(self,videoId):
        return self._videos[videoId].isOpen()

    def currentVideoFrame(self,videoId):
        return self._videos[videoId].currentFrame()
    
    def videoLength(self,videoId):
        return self._videos[videoId].numOfFrames()
    
    def videoNext(self,videoId,jump=1):
        return self._videos[videoId].readNext(jump)
    
    def videoPrev(self,videoId,jump=1):
        return self._videos[videoId].readPrev(jump)

    def videoCurrent(self,videoId):
        return self._videos[videoId].read()
    
    def videoGoto(self,videoId,goto):
        return self._videos[videoId].goto(goto)

    def sampleFrame(self,videoId):
        video = self._videos[videoId]
        frame = video.currentFrame()
        name = f"{videoId}_{frame}"
        if name in self._itemNames:
            itemid = self._items[name].id()
            overwrite = notify(f"{name} already exists. Do you want to overwrite it?","yesno")
            if not overwrite:
                return False,name,itemid    
        else:
            self._itemNames.append(name)
            itemid = len(self._itemNames)-1
        
        ret,frame = video.read()
        if ret:
            imageio.imwrite(f"{self._path}/imgs/{name}.jpg", frame)
            self._items[name] = DatasetItem.create(self._path,name,f"{name}.jpg",itemid) 

        return True,name,itemid


    @classmethod
    def load(cls,path):
        imgPath = f"{path}/imgs/"
        annotationPath = f"{path}/annotations/"
        maskPath = f"{path}/masks/"
        keysPath = f"{path}/keys/"
        keysColorFile = f"{path}/keys/keys.json"
        videosPath = f"{path}/videos"

        if not exists(imgPath) or not exists(annotationPath) or not exists(keysPath) or not exists(maskPath):
            return False,None,"Dataset folder must contain four folders: imgs, annotations, masks, and keys"
        
        imgs = sorted([f for f in os.listdir(imgPath) if f.upper().endswith(VALID_FORMAT)])

        keys = sorted([f for f in os.listdir(keysPath) if f.upper().endswith(VALID_FORMAT)])
        if len(keys) == 0:
            return False,None,"The dataset doesn't contain any key"
        
        if os.path.exists(videosPath):
            videos = sorted([f for f in os.listdir(videosPath) if f.upper().endswith(VALID_VIDEO_FORMAT)])
        else:
            videos = []

        return True, Dataset(path,imgs,keys,videos),""
