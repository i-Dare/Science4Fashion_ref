import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
#Magic Numbers
BORDER = 0.25

def grabcut(image, mask = None, times = 50):
    ''' Applying Grabcut algorithm, which extracts pretty good the foreground for one color image. In case a mask is given as parameter it uses it as the starting mask for the background.

        Reference
        ----------
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html

        Parameters
        ----------
        image : array-like(image), shape [M, N, 3]
                RGB/BGR format image.

        mask : array-like(image), shape [M, N]
                Binary color format, 0 for background and 255 for foreground.
        times : int
            number of iteration the algorithm ran.

        Returns
        -------
        mask : array-like(image), shape [M, N]
            boolean format, 0 for background and 1 for foreground.
    '''
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    if mask is None:
        mask = np.zeros(image.shape[:2],np.uint8)
        #we can change the indexes. It is like (x_start, y_start,x_end, y_end)
        rect = (1,1,image.shape[1]-1,image.shape[0]-1)
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, times, cv2.GC_INIT_WITH_RECT)
    else:
        kernel = np.ones((50, 50), np.float32) / (50 * 50)
        dst = cv2.filter2D(mask, -1, kernel)
        dst[dst != 0] = 255
        free = np.array(cv2.bitwise_not(dst), dtype=np.uint8)
        grab_mask = np.zeros(mask.shape, dtype=np.uint8)
        grab_mask[:, :] = 2
        grab_mask[mask == 255] = 1
        grab_mask[free == 255] = 0
        # mask, _, _ = cv2.grabCut(image, grab_mask, None, bgdModel, fgdModel, times, cv2.GC_INIT_WITH_MASK)  
        

    # mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # return mask
    return grab_mask

def closing(mask):
    ''' Creating closing element and applying it to the calculated mask.

        Parameters
        ----------
        mask: array-like(image), shape [M, N]
                boolean format, 0 for other and 1 for skin.

        Returns
        -------
        mask: array-like(image), shape [M, N]
                boolean format, 0 for other and 1 for skin.
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask



def reshapeDim(mask2D, image3D):
    '''Reshaping both mask and image to degenerate the first two dimensions to one.
        Parameters
        ----------
        mask2D : array-like(image), shape [M, N]
                boolean format, 0 for background and 1 for foreground.
        image3D : array-like(image), shape [M, N, 3]
                RGB format image, that have been masked by the grabcut algorithm mask getting rid of background.

        Returns
        -------
        mask1D : array-like(image), shape [M*N, 3]
                RGB format image, without the background entries.
        image2D : array-like(image), shape [M*N]
                boolean format, False and True to filter the entries of image
        '''
    #reshape from (Μ,Ν,3) to (Μ*Ν,3)
    image2D = image3D.reshape((image3D.shape[0] * image3D.shape[1], 3))
    #Choosing only the masked elements, without taking into consideration the background
    mask1D = mask2D.reshape((mask2D.shape[0] * mask2D.shape[1]))
    # mask1D = mask1D == 255 if (mask1D == 1).sum()==0 else mask1D == 1

    mask1D = mask1D == 1
    image2D = image2D[mask1D, :]
    return mask1D, image2D

def colorRectangle(colors, sizeImage):
    ''' Constructs the rectangle in which the different colors included in the image will be displayed. 

        Reference
        ----------

        Part of the implementation was found at https://stackoverflow.com/questions/28793985/find-dominant-color-on-an-image

        Parameters
        ----------
        colors: list of tuples, shape Nx(percent of color(decimal), [R,G,B])
        sizeImage : int, dimension of image
                    the M of [MxNxC]

        Returns
        -------
        rect : numpy array, shape [sizeImage, BORDER, 3]
            Rectangle containing the detected colors of the image.

    '''
    sizeY = int(BORDER*sizeImage)
    rect = np.zeros((sizeImage, sizeY, 3), dtype=np.uint8)
    start = 0
    for (percent, color) in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * sizeImage)
        cv2.rectangle(rect, (0, int(start)), (sizeY, int(end)), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect

def rectImage(imgRGB, rectVisual):
    ''' Merge image with the color palette. 

        Parameters
        ----------
        imgRGB : array-like(image), shape [sizeImage, N, 3]
                RGB format initial-image.
        rectVisual : numpy array, shape [sizeImage, BORDER, 3]
            Rectangle containing the n_clusters detector colors of the image.

        Returns
        -------
        visual : numpy array(image), shape [sizeImage, BORDER + N, 3]
            Image containing the rectangle and the RGB format image
    '''
    visual = np.zeros((imgRGB.shape[0], rectVisual.shape[1] + imgRGB.shape[1], 3), dtype=np.uint8)
    visual[0:imgRGB.shape[0], 0:rectVisual.shape[1], :] = rectVisual
    visual[0:imgRGB.shape[0], rectVisual.shape[1]:(imgRGB.shape[1]+rectVisual.shape[1]), :] = imgRGB
    return visual


def visualizeGrid(images):
    ''' Merge image with the color palette. 

        Parameters
        ----------
        images : list of image
                Contains the images [imageRGB, mask, imageCut, visual]
        Returns
        -------
        nothing : displays grid of images (2 rows, columns)
    '''
    size = len(images)
    fig = plt.figure(figsize=(15, 7))
    columns = int(math.ceil(size/2))
    rows = 2
    for i in range(1, size + 1):
        img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show() 