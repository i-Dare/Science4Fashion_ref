# -*- coding: utf-8 -*-
#For code
from PIL import Image
from sklearn.cluster import KMeans
import imageUtils
import segmentation
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#for class
import numpy as np
import cv2



class Cloth:
    '''Every cloth attribute will be included in this class as well as any helpful function to extract the relevant attributes.
    
    '''
    def __init__(self, path = None, human = 0, imgBGR = None, segBackgroundMask = None, catsSegment = 0, skinMask = None, clothMask = None, clothImg = None, colors = None):
        self.path = path
        self.human = human
        # Check if image is in grayscale and convert to BGR
        if len(imgBGR.shape) < 3:
            self.imgBGR = cv2.cvtColor(imgBGR, cv2.COLOR_GRAY2BGR)
        else:
            self.imgBGR = imgBGR
        
        self.imgRGB = cv2.cvtColor(self.imgBGR, cv2.COLOR_BGR2RGB)            
        self.segBackgroundMask = segBackgroundMask
        self.catsSegment = catsSegment
        self.skinMask = skinMask
        self.clothMask = clothMask
        self.clothImg = clothImg
        self.colors = colors

    def __call__(self, path = None, human = 0, imgBGR = None, segBackgroundMask = None, catsSegment = 0, skinMask = None, clothMask = None, clothImg = None, colors = None):
        self.path = path
        self.human = human
        self.imgBGR = imgBGR
        self.imgRGB = cv2.cvtColor(self.imgBGR, cv2.COLOR_BGR2RGB)
        self.segBackgroundMask = segBackgroundMask
        self.catsSegment = catsSegment
        self.skinMask = skinMask
        self.clothMask = clothMask
        self.clothImg = clothImg
        self.colors = colors

    def PILtoBGR(self, img):
        ''' Converting PIL image to BGR format.

            Parameters
            ----------
            img : array-like(image), shape [M, N, 3]
                    PIL format image.

            Returns(Inside-Class)
            -------
            img : array-like(image), shape [M, N, 3]
                    BGR format image.
        '''
        imageRGB = np.array(img)
        self.imgBGR = imageRGB[:, :, ::-1]

    def discardExtraCats(self):
        ''' In case of recognizing another class except background or person, the background mask that has been segmented it is passed through grabcut algorithm in order to refine it.

            Parameters(Inside-Class)
            ----------
            imgBGR : array-like(image), shape [M, N, 3]
                    BGR format image.
            catsSegment : int
                    Count of classes that have been recognized.
            segBackgroundMask : array-like(image), shape [M, N]
                    segmented format, a number for each class in FULL_LABEL_MAP.

            Returns(Inside-Class)
            -------
            segBackgroundMask : array-like(image), shape [M, N]
                segmented format, a number for each class in FULL_LABEL_MAP.
        '''
        if len(self.catsSegment) > 2:
            self.segBackgroundMask = np.where(self.segBackgroundMask == 15, 255, 0)
            self.segBackgroundMask = imageUtils.grabcut(self.imgBGR, self.segBackgroundMask)
        else:
            self.segBackgroundMask = np.where(self.segBackgroundMask == 15, 1, 0)
    
    def get_hsv_mask(self):
        ''' Calculating skin mask using HSV color space.

            Reference
            ----------
            Human Skin Detection Using RGB, HSV and YCbCr Color Models
            https://arxiv.org/pdf/1708.02694.pdf

            Parameters(Inside-Class)
            ----------
            img : array-like(image), shape [M, N, 3]
                    BGR format image.

            Returns
            -------
            msk_hsv : array-like(image), shape [M, N]
                    boolean format, 0 for other and 1 for skin.
        '''
        lower_thresh = np.array([0, 58, 0], dtype=np.uint8)
        upper_thresh = np.array([25, 173, 255], dtype=np.uint8)
        img_hsv = cv2.cvtColor(self.imgBGR, cv2.COLOR_BGR2HSV)
        msk_hsv = cv2.inRange(img_hsv, lower_thresh, upper_thresh)
        msk_hsv[msk_hsv < 128] = 0
        msk_hsv[msk_hsv >= 128] = 1
        msk_hsv = msk_hsv.astype(float)
        return msk_hsv

    def get_rgb_mask(self):
        ''' Calculating skin mask using RGB color space.

            Reference
            ----------
            Human Skin Detection Using RGB, HSV and YCbCr Color Models
            https://arxiv.org/pdf/1708.02694.pdf

            Parameters(Inside-Class)
            ----------
            img : array-like(image), shape [M, N, 3]
                    BGR format image.

            Returns
            -------
            msk_rgb : array-like(image), shape [M, N]
                    boolean format, 0 for other and 1 for skin.
        '''
        lower_thresh = np.array([95, 40, 20], dtype=np.uint8)
        upper_thresh = np.array([255, 255, 255], dtype=np.uint8)
        (R, G, B) = cv2.split(self.imgRGB)
        msk_a = cv2.inRange(self.imgRGB, lower_thresh, upper_thresh)
        msk_b = (R > G).astype(int)*255
        msk_c = (R > B).astype(int)*255
        msk_d = (R > (G + 15)).astype(int)*255
        msk_e = np.bitwise_and(np.uint64(msk_a), np.uint64(msk_b))
        msk_f = np.bitwise_and(np.uint64(msk_c), np.uint64(msk_d))
        msk_rgb = np.bitwise_and(np.uint64(msk_e), np.uint64(msk_f))
        msk_rgb[msk_rgb < 128] = 0
        msk_rgb[msk_rgb >= 128] = 1
        msk_rgb = msk_rgb.astype(float)
        return msk_rgb

    def get_ycrcb_mask(self):
        ''' Calculating skin mask using YCrCb color space.

            Reference
            ----------
            https://fiveko.com/tutorials/image-processing/skin-detection-segmentation/
            FivekoGFX implementation
            http://www.wseas.us/e-library/conferences/2011/Mexico/CEMATH/CEMATH-20.pdf

            https://nalinc.github.io/blog/2018/skin-detection-python-opencv/

            Parameters(Inside-Class)
            ----------
            img : array-like(image), shape [M, N, 3]
                    BGR format image.

            Returns
            -------
            msk_ycrcb : array-like(image), shape [M, N]
                    boolean format, 0 for other and 1 for skin.
        '''
        lower_thresh = np.array([0, 133, 80], dtype=np.uint8)
        upper_thresh = np.array([255, 173, 120], dtype=np.uint8)
        img_ycrcb = cv2.cvtColor(self.imgBGR, cv2.COLOR_BGR2YCR_CB)
        msk_ycrcb = cv2.inRange(img_ycrcb, lower_thresh, upper_thresh)
        msk_ycrcb[msk_ycrcb < 128] = 0
        msk_ycrcb[msk_ycrcb >= 128] = 1
        msk_ycrcb = msk_ycrcb.astype(float)
        return msk_ycrcb

    def skinExtraction(self, thresh=0.5):  
        ''' Skin mask extraction process.

            Parameters
            ----------
            thresh: float
                    Represents the threshold to determine foreground and background
            Returns
            -------
            skinMask: array-like(image), shape [M, N]
                    boolean format, 1 for other and 0 for skin.
        '''
        msk_hsv = self.get_hsv_mask()
        msk_rgb = self.get_rgb_mask()
        msk_ycrcb = self.get_ycrcb_mask()
        n_masks = 2.0
        mask = (msk_hsv + msk_ycrcb) / n_masks
        mask_blured = cv2.blur(mask,(10,10))
        mask_dilated = cv2.dilate(mask_blured, (5,5), iterations=3)
        mask = cv2.threshold(mask_dilated, 0.5, 255, cv2.THRESH_BINARY)[1]
        mask = mask.astype('uint8')
        maskclosing = imageUtils.closing(mask)
        self.skinMask = 1 - maskclosing
    
    def combineMasks(self):
        ''' Extracting the cloth in RGB format, using the mask containing background and skin.

            Returns(Inside-Class)
            -------
            clothImg: array-like(image), shape [M, N, 3]
                    RGB format
        '''
        # Attemp Canny edge detection
        imgGray = cv2.cvtColor(self.imgBGR, cv2.COLOR_BGR2GRAY)
        imgCanny = cv2.Canny(imgGray, threshold1=1, threshold2=50)
        # For debug purposes
        self.imgCanny = imgCanny.copy()
        # Attempt to remove background and skin
        if self.segBackgroundMask is not None:
            if self.segBackgroundMask.sum() > 0:
                dim = (self.imgBGR.shape[1], self.imgBGR.shape[0])
                resized_segBackgroundMask = cv2.resize(self.segBackgroundMask, dim, interpolation = cv2.INTER_NEAREST)
                self.clothMask = resized_segBackgroundMask & self.skinMask 
            else:
                imgCanny_c = imgCanny.copy()
                _, contours, _ = cv2.findContours(imgCanny_c, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                imgZeros = np.zeros_like(imgCanny_c)
                cv2.drawContours(imgZeros, contours, -1, 255, 100)
                kernel = np.ones((5,5), np.uint8)
                imgZeros = cv2.dilate(imgZeros, kernel, iterations=3)
                self.clothMask = cv2.threshold(imgZeros, 0.5, 1, cv2.THRESH_BINARY)[1]
        else:         
            self.clothMask = imageUtils.grabcut(self.imgBGR, imgCanny, 10)
        imgRGB = cv2.cvtColor(self.imgBGR, cv2.COLOR_BGR2RGB)
        self.clothImg = imgRGB * self.clothMask[:, :, np.newaxis]

    def extractColor(self, img2D, numCluster = 5):
        ''' Extracting color and percentage of each color.
            Reference
            ----------

            Part of the implementation was found at https://stackoverflow.com/questions/28793985/find-dominant-color-on-an-image

            Parameters
            ----------
            img2D : array-like(image), shape [M*N]
                boolean format, False and True to filter the entries of image
            numClusters : int
                        number of centers of cluster
            
            Returns(Inside-Class)
                -------
                colors: list of tuples, shape Nx(percent of color(decimal), [R,G,B])
        '''
        try:
            cluster = KMeans(n_clusters=numCluster).fit(img2D)
            # Get the number of different clusters, create histogram, and normalize
            labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
            (hist, _) = np.histogram(cluster.labels_, bins = labels)
            hist = hist.astype("float")
            hist /= hist.sum()
            # Create frequency rect and iterate through each cluster's color and percentage
            self.colors = sorted([(percent, np.round(color).astype(int)) for (percent, color) in zip(hist, cluster.cluster_centers_)], reverse = True)
        except:
            temp = -1*np.ones(3,dtype = int)
            self.colors = [(0.,temp),(0.,temp),(0.,temp),(0.,temp),(0.,temp)]

if __name__ == "__main__":
    # imgPath = 'D:\\Databases\\Energiers Images/PHOTOS ΠΑΝΕΠΙΣΤΗΜΙΟ/SUMMER 2018/photos ss 2018 site/energiers/11-218450-0_013-01.jpg'
    imgPath = 'D:/Databases/Images/PHOTOS ΠΑΝΕΠΙΣΤΗΜΙΟ/EXTRA/15-219361-2.jpg'
    imgCloth = Cloth(imgPath, 0)
    if imgCloth.human:
        modelPath = 'C:/Users\sotiris/OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης/Σωτήρης/Science4Fashion/Skin Model/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
        #Segmentation Background & Person
        LABEL_NAMES = np.asarray([
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv' 
        ])

        FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
        FULL_COLOR_MAP = segmentation.label_to_color_image(FULL_LABEL_MAP)
        odapi = segmentation.DeepLabModel(tarball_path=modelPath)
        imgPIL = Image.open(imgPath)
        imgPIL, imgCloth.segBackgroundMask, imgCloth.catsSegment = odapi.run(imgPIL)
        #End-Segmentation Background & Person
        #--------------------------------------------------------------------------------#
        #Extraction Mask of Clothes
        imgCloth.PILtoBGR(imgPIL)
        imgCloth.discardExtraCats()
        imgCloth.skinExtraction()
        imgCloth.combineMasks()
        #End-Extraction Mask of Clothes
        #--------------------------------------------------------------------------------#
        _, clothImg2D = imageUtils.reshapeDim(imgCloth.clothMask, imgCloth.clothImg)
        # Find and display most dominant colors
        #cluster = KMeans(n_clusters=5).fit(clothImg2D)
        imgCloth.extractColor(clothImg2D)

        rectVisual = imageUtils.colorRectangle(imgCloth.colors, imgCloth.clothImg.shape[0])
        imgRGB = cv2.cvtColor(imgCloth.imgBGR, cv2.COLOR_BGR2RGB)
        visual = imageUtils.rectImage(imgRGB, rectVisual)
        images = [imgRGB, imgCloth.segBackgroundMask, imgCloth.skinMask, imgCloth.clothMask, imgCloth.clothImg, visual]
        imageUtils.visualizeGrid(images)

    else:
        #imgCloth.imgBGR = cv2.imread(imgPath)
        #solution for greek path names
        f = open(imgPath, "rb")
        chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        imgCloth.imgBGR = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        #end try
        imgCloth.clothMask = imageUtils.grabcut(imgCloth.imgBGR, None, 10)
        imgRGB = cv2.cvtColor(imgCloth.imgBGR, cv2.COLOR_BGR2RGB)
        imgCloth.clothImg = imgRGB*imgCloth.clothMask[:, :, np.newaxis]
        #Add the color palette
        _, clothImg2D = imageUtils.reshapeDim(imgCloth.clothMask, imgCloth.clothImg)
        # Find and display most dominant colors
        imgCloth.extractColor(clothImg2D)
        rectVisual = imageUtils.colorRectangle(imgCloth.colors, imgCloth.clothImg.shape[0])
        visual = imageUtils.rectImage(imgRGB, rectVisual)
        images = [imgRGB, imgCloth.clothMask, imgCloth.clothImg, visual]
        imageUtils.visualizeGrid(images)
        