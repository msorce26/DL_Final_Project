
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


filename = 'Latitude.csv'
#REMEMBER TO CHANGE THESE NUMBER EVERY TIME YOU RUN
upperrangeofImages = 2000
lowerrangeofImages = 1999
NumTimesRan = 3

temp = ("cropped_data" + str(NumTimesRan) + ".xls")


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location) 
            
#Converts a JPG image to an array
def jpg_image_to_array(image_path):
  """
  Loads JPEG image into 3D Numpy array of shape 
  (width, height, channels)
  """
  with Image.open(image_path) as image:         
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
  return im_arr

#Used to unroll a List of List into a single List
def unroll2(images, values):
    flat_list = []
    y_list = []
    i = 0
    for sublist in images:
        for item in sublist:
            flat_list.append(item)
            y_list.append(values[i])
        i = i + 1
    return flat_list, y_list

#Gets the first element in a list thats in a list
def getFirstElement(array):
    xdata = []
    
    for file in array:
        xdata.append(file[0])
    
    return xdata

#Converts and RGB array into a grayscale Array
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#######################################################################
#Imports the CSV and separates the feature data from the labels
import csv
 
image_path = []
value = []
 
with open(filename) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        image_path.append(row[0])
        value.append(row[1])

del image_path[0]
del value[0]
#######################################################################
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
#
#xdata, ydata = unroll2(imagefile_path, labels)
xdata = getFirstElement(imagefile_path)
        
xdat = xdata[lowerrangeofImages:upperrangeofImages]
i = 0
import xlwt

book = xlwt.Workbook(encoding="utf-8")
sh = book.add_sheet("Sheet ")
    
for image in xdat:
    img_path = image.replace(".jpg","")
    img_path = img_path.replace("Latitude(deg)","Latitude_Individ")
    
    crop_path = []
    crop_path.append((img_path + 'crop_1.jpg'))
    crop_path.append((img_path + 'crop_2.jpg'))
    crop_path.append((img_path + 'crop_3.jpg'))
    
    ###########################################################
    #If the number is not CENTERED, THIS IS WHAT YOU NEED TO CHANGE TO CENTER IT
    
    crop(image,(5,6,10,13), crop_path[0])
    crop(image,(10,6,15,13), crop_path[1])
    crop(image,(15,6,20,13), crop_path[2])
    
    
    import matplotlib.image as mpimg
    img=mpimg.imread(image)
    imgplot = plt.imshow(img)
    plt.show()
    
    img=mpimg.imread(crop_path[0])
    imgplot = plt.imshow(img)
    plt.show()
    
    img=mpimg.imread(crop_path[1])
    imgplot = plt.imshow(img)
    plt.show()
    
    img=mpimg.imread(crop_path[2])
    imgplot = plt.imshow(img)
    plt.show()
    
    num = input("#,#,#")
    print(num)
    
    crop_lab = num.split(",")
    #####################################3
    
    for path,label in zip(crop_path, crop_lab):
        i = i+1
        sh.write(i, 0, path)
        sh.write(i, 1, label)

#for lat, long in zip(Latitudes, Longitudes):
#    print lat, long
    book.save(temp)

