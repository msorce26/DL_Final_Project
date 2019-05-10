# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:58:11 2019

@author: td-adm
"""

from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

model = load_model('my_model1.h5')
filename = 'Latitude.csv'
upperrangeofImages = 2504
lowerrangeofImages = 2503

def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location) 
    
#Gets the first element in a list thats in a list
def getFirstElement(array):
    xdata = []
    
    for file in array:
        xdata.append(file[0])
    
    return xdata
    
def jpg_image_to_array(image_path):
  """
  Loads JPEG image into 3D Numpy array of shape 
  (width, height, channels)
  """
  with Image.open(image_path) as image:         
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
  return im_arr

###################################################################
import csv
 
image_path = []
value = []
 
with open(filename) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        image_path.append(row[0])
        value.append(row[1])

###################################################################
#Seperates the feature data because there are multiple images in each entry
filepath_array = []        
for string in range(len(image_path)):
    removedComma = image_path[string].replace(",","")
    removeasf = removedComma.replace(".asf","")
    text = removeasf.split("./")
    del text[0]
    filepath_array.append(text)     
    
#######################################################################
    
#Removes any entry that has no path to a picture
imagefile_path = []
labels = []
for index in range(len(image_path)):
    if image_path[index]:
        imagefile_path.append(filepath_array[index])
        labels.append(value[index])
        
xdata = getFirstElement(imagefile_path)
        
xdat = xdata[lowerrangeofImages:upperrangeofImages]



for image in xdat:
    img_path = image.replace(".jpg","")
    img_path = img_path.replace("Latitude(deg)","Latitude_Individ")
    
    crop_path = []
    crop_path.append((img_path + 'crop_1.jpg'))
    crop_path.append((img_path + 'crop_2.jpg'))
    crop_path.append((img_path + 'crop_3.jpg'))
    
    ###########################################################
    #If the number is not CENTERED, THIS IS WHAT YOU NEED TO CHANGE TO CENTER IT
    
    crop(image,(5,7,10,14), crop_path[0])
    crop(image,(10,7,15,14), crop_path[1])
    crop(image,(15,7,20,14), crop_path[2])
    
    y = []
    images = []
    
    import matplotlib.image as mpimg
    img=mpimg.imread(image)
    imgplot = plt.imshow(img)
    plt.show()
    
    img=mpimg.imread(crop_path[0])
    images.append(img)
    imgplot = plt.imshow(img)
    plt.show()
    
    img=mpimg.imread(crop_path[1])
    images.append(img)
    imgplot = plt.imshow(img)
    plt.show()

    img=mpimg.imread(crop_path[2])
    images.append(img)
    imgplot = plt.imshow(img)
    plt.show()

    images = zoom(images,(1,5,7,1))
    y.append(model.predict(images, batch_size=None, verbose=0,steps=None))

    print(str(y))
        
        
#    model.predict(x, batch_size=None, verbose=0,steps=None,callbacks=None)
