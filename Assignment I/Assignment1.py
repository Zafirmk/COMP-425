import cv2
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt


#Function to downsample an image by a given factor
def downsample(image, factor, colored=False):
    # Store values of original width, height and number of color channels
    originalWidth = image.shape[0]
    originalHeight = image.shape[1]
    if colored:
        colorChannels = image.shape[2]
    
    # Calculate new width and new height of downsampled image
    newWidth = originalWidth // factor
    newHeight = originalHeight // factor
    
    # Initialize empty image
    if colored:
        newImage = np.zeros((newWidth, newHeight, colorChannels)).astype('uint8')
    else:
        newImage = np.zeros((newWidth, newHeight)).astype('uint8')
    
    # Iterate over the new image row and column wise
    for i in range(0, newWidth):
        for j in range(0, newHeight):
            
            # Set the pixel i,j of new image to i*factor and j*factor from original image
            newImage[i,j] = image[i*factor,j*factor]
    return newImage

# Function to display the downsample images one by one
def displayImages(image, factors, colored=False):
    for factor in factors:
        cv2.imshow(str(factor) + "x Downsample", downsample(image, factor, colored=colored))
        cv2.waitKey(0)

# Function to calculate amount of padding to be added based on a given kernel 
def calculatePadding(kernel):
    kernelRows = kernel.shape[0]    
    return(kernelRows-1)

# Function to calculate the output size of an image (with or without padding)
def calculateOutputSize(image, kernel, padded):
    
    # Store image size
    rows = image.shape[0]
    cols = image.shape[1]
    
    # Store kernel size
    kernelWidth = kernel.shape[0]
    kernelHeight = kernel.shape[1]
    
    # Calculate padding amount that needs to be added based on kernel shape
    paddingAmount = calculatePadding(kernel)
    
    # Calculate new image size based on whether or not image is padded
    if padded:
        width = int(rows-kernelWidth + paddingAmount) + 1
        height = int(cols-kernelHeight + paddingAmount) + 1
    else:
        width = int(rows-kernelWidth) + 1
        height = int(cols-kernelHeight) + 1
    
    return ((width, height))

# Function to add padding onto the image (colored or b&w)
def addPadding(image, kernel, colored=False):
    
    # Store dimensions of image
    rows = image.shape[0]
    cols = image.shape[1]
    paddingAmount = calculatePadding(kernel)
    
    # Create padded image with 3 color channels for colored image and 1 channel for b&w image
    if colored:
        paddedImage = np.zeros((rows+(2*paddingAmount), cols+(2*paddingAmount), 3))
        paddedImage[paddingAmount:-paddingAmount, paddingAmount:-paddingAmount] = image

    else:
        paddedImage = np.zeros((rows+(2*paddingAmount), cols+(2*paddingAmount)))
        paddedImage[paddingAmount:-paddingAmount, paddingAmount:-paddingAmount] = image
    
    return (paddedImage)

# Function to pass a given kernel over a colored or b&w (padded or non padded) image
def convolve(image, kernel, padding=False, colored=False):
    
    if len(image.shape) == 3 and colored == False:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Store dimensions of kernel and image
    kernelWidth = kernel.shape[0]
    kernelHeight = kernel.shape[1]
    
    rows = image.shape[0]
    cols = image.shape[1]
    
    # Calculate size of output image
    outputWidth, outputHeight = calculateOutputSize(image, kernel, padding)
    
    # Add padding to the image if required
    if padding:
        paddedImage = addPadding(image, kernel, colored)
    else:
        paddedImage = image
    
    # Seperate color channels for colored image, else single channel for b&w image  
    if colored:
        c = 3
        outR = np.zeros((outputWidth, outputHeight)).astype('uint8')
        outG = np.zeros((outputWidth, outputHeight)).astype('uint8')
        outB = np.zeros((outputWidth, outputHeight)).astype('uint8')
    else:
        c = 1
        out = np.zeros((outputWidth, outputHeight))

    # Loop over for every color
    for k in range(c):
        
        # Loop over every row
        for x in range(rows):
            
            # Check if row is in range of the image
            if x + kernelWidth - padding <= rows:
                
                # Loop over every coloumn
                for y in range(cols):
                    
                    # Check if coloumn is in range of the image
                    if y + kernelHeight - padding <= cols:
                        
                        # Obtain neighborhood reigon
                        if colored:
                            neighborhood = paddedImage[x:x+kernelWidth, y:y+kernelWidth, k]
                        else:
                            neighborhood = paddedImage[x:x+kernelWidth, y:y+kernelWidth]
                        
                        # Multiply kernel with neighborhood and normalize finalVal
                        finalVal = (kernel*neighborhood).sum()
                        if finalVal < 0:
                            finalVal = 0
                        elif finalVal > 255:
                            finalVal = 255
                        
                        # Add finalVal to correct color channel for colored image
                        if colored:
                            if k == 0:
                                outR[x,y] = finalVal
                            elif k == 1:
                                outG[x,y] = finalVal
                            else:
                                outB[x,y] = finalVal
                        else:
                            out[x,y] = finalVal
    
    # If colored combine all color channels and return
    if colored:
        return np.dstack((outR,outG,outB))
    else:
        return out.astype("uint8")

# Function to shift image diagonally towards top right 
def topRightShift(image, colored, iterations):
    kernel = np.array([[0,0,0],[0,0,0],[1,0,0]])
    
    out = image
    
    print("\nOriginal Image: " + str(out.shape))
    
    for i in range(iterations):
        print("\nNow processing iteration number " + str(i+1))
        out = convolve(out, kernel, False, colored=colored)
    
    # Reapply padding on the left and bottom
    
    print("\nAfter Shift: " + str(out.shape))
    if colored:
        out = np.pad(out, ((0,iterations*2),(iterations*2, 0), (0,0)), "constant")
    else:
        out = np.pad(out, ((0,iterations*2),(iterations*2, 0)), "constant")
    print("\nAfter Padding: " + str(out.shape))
    
    return(out)

# Function to output gaussian matrix
def gaussian(size, sigma):
    
    # Fix size of gaussian matrix
    filter_size = size
    
    # Initialize empty matrix
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2
    
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2

    # Normalize values and return
    return gaussian_filter/np.sum(gaussian_filter)

# Function to calculate difference of two gaussian filters applied to colored or b&w (padded or non padded) image
def gaussianDifference(image, ksize, alpha, beta, padding=False, colored=False):
    f1 = gaussian(ksize, alpha)
    f2 = gaussian(ksize, beta)
    
    a = convolve(image, f1, padding, colored)
    b = convolve(image, f2, padding, colored)
    
    return np.absolute(a-b)
        
# Function to calculate the Sobel X of an image
def sobelX(image, padding=False, colored=False):
    filter = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    return(convolve(image, filter, padding, colored))

# Function to calculate the Sobel Y of an image
def sobelY(image, padding=False, colored=False):
    filter = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])    
    return(convolve(image, filter, padding, colored))

# Function to calculate orientation map of a given colored or b&w (padded or non padded) 
def pixelOrientation(image, padding=False, colored=False):
    return((np.degrees(np.arctan2(sobelY(image, padding, colored), sobelX(image, padding, colored)))).astype('uint8'))

# Function to calculate magnitude map of a given colored or b&w (padded or non padded) image
def magnitudeMap(image, padding=False, colored=False):
    return((np.hypot(sobelX(image, padding, colored), sobelY(image, padding, colored))).astype('uint8'))

# Function to return cv2 canny applied to an image
def canny(image, t1, t2):
    return(cv2.Canny(image, t1, t2))

# Function to threshold from the magnitude map (2nd step of Canny edge detection)
def thresholdingInitial(image, threshold, padding=False):

    # Same process followed as convolve
    
    filterX = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    filterY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    kernelWidth = filterX.shape[0]
    kernelHeight = filterX.shape[1]
    
    rows = image.shape[0]
    cols = image.shape[1]
    
    outputWidth, outputHeight = calculateOutputSize(image, filterX, padding)
    
    if padding:
        paddedImage = addPadding(image, filterX, False)
    else:
        paddedImage = image
        

    out = np.zeros((outputWidth, outputHeight)).astype('uint8')    

    for x in range(rows):
        if x + kernelWidth - padding <= rows:
            for y in range(cols):
                if y + kernelHeight - padding <= cols:
                    finalValX = int((filterX*paddedImage[x:x+kernelWidth, y:y+kernelWidth]).sum())
                    finalValY = int((filterY*paddedImage[x:x+kernelWidth, y:y+kernelWidth]).sum())
                    
                    # New valToAdd calculated to get magnitude of each pixel
                    valToAdd = np.hypot(finalValX, finalValY)
                    
                    # New valToAdd must be above a given threshold (Tm)
                    if valToAdd > threshold:
                        out[x,y] = valToAdd
    return out

# Function to perform non max surpression (3rd step of Canny edge detection)
def nonMaxSuppresion(img):
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    orientationMap = pixelOrientation(img)
    magmap = magnitudeMap(img)
        
    rows = orientationMap.shape[0]
    cols = orientationMap.shape[1]
    out = np.zeros_like(img)
    
    for i in range(0, rows):
        for j in range(0, cols):
            
            # Check gradient direction for each pixel
            # If pixels in given gradient direction are more than the current pixel
            # Replace its value with the value from magmap
            try:
                a = 255
                b = 255
                
                if 0 <= orientationMap[i,j] < 22.5:
                    a = magmap[i, j+1]
                    b = magmap[i, j-1]
                elif 22.5 <= orientationMap[i,j] < 67.5:
                    a = magmap[i+1, j-1]
                    b = magmap[i-1, j+1]
                elif 67.5 <= orientationMap[i,j] < 112.5:
                    a = magmap[i+1, j]
                    b = magmap[i-1, j]
                elif 112.5 <= orientationMap[i,j] < 157.5:
                    a = magmap[i-1, j-1]
                    b = magmap[i+1, j+1]
                if (magmap[i,j] >= a) and (magmap[i,j] >= b):
                    out[i,j] = magmap[i,j]
                else:
                    out[i,j] = 0
                
            except IndexError as e:
                pass          
            
    return out

# Function to keep edge segments (4.1 step of Canny edge detection)
def thresholdLatter(img, low, high):
    
    # Secondary thresholding where we check if a pixel is strong, weak or labeled as 0
    weak = low
    strong = high
    
    highThreshold = strong * 0.09;
    lowThreshold = weak * 0.05;
    
    out = np.zeros((img.shape[0],img.shape[1])).astype('uint8')
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            
            if img[i, j] >= highThreshold:
                out[i, j] = strong
            elif (img[i, j] <= highThreshold) & (img[i,j] >= lowThreshold):
                out[i, j] = weak
            else:
                out[i,j] = 0
    
    return (out)

# Function to define weak edges in second thresholding (4.2 step of canny edge detection)
def hysteresis(img, weak, strong):
    rows = img.shape[0]
    cols = img.shape[1]
    
    # For any pixel labeled weak, check the reigon around it to see if it has a strong pixel
    # If it does then make the weak pixel a strong pixel
    for i in range(0, rows):
        for j in range(0, cols):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

# Function to run custom canny edge detection
def customCanny(img, t1, t2, tm):
    
    # Blur Image first (1)
    print("\nCustom Canny Edge Detection    -   Gaussian Blurring (1)")
    gaussianBlurred_img = convolve(img, gaussian(3, 1))
    
    # Initial Thresholding of the magnitude map (2)
    print("\nCustom Canny Edge Detection    -   Magnitude Map & Thresholding Magnitude (2)")
    initialThreshold = thresholdingInitial(gaussianBlurred_img, tm)

    # Apply Non Max Suppression (3)
    print("\nCustom Canny Edge Detection    -   Non Max Suppression (3)")
    nonMaxSuppressed = nonMaxSuppresion(initialThreshold)

    # Apply second thresholding followed by Hysteresis (4, 5)
    print("\nCustom Canny Edge Detection    -   Second Thresholding (4)")
    thresholdedLatter = thresholdLatter(nonMaxSuppressed, t1, t2)

    print("\nCustom Canny Edge Detection    -   Hysteresis (5)")
    final = hysteresis(thresholdedLatter, t1, t2)
    
    return final.astype('uint8')

# Function to download CIFAR10 Dataset
def downloadCIFAR():
    return torchvision.datasets.CIFAR10(root="./data", download=True, train=True)

# Function to display CIFAR10 Dataset images
def displayCIFAR():
    
    # Download CIFAR training data set
    train_set = list(downloadCIFAR())
    imagesToDisplay = []
    classesAdded = []
    
    # Labels of each image type
    mydict = {0:'airplane',
            1:'automobile',
            2:'bird',
            3:'cat',
            4:'deer',
            5:'dog',
            6:'frog',
            7:'horse',
            8:'ship',
            9:'truck'}
    
    # Select 10 random images
    while len(imagesToDisplay) < 10:
        num = random.randint(0, len(train_set)-1)
        currPic = train_set[num]
        if currPic[1] in classesAdded:
            pass
        else:
            imagesToDisplay.append(currPic)
            classesAdded.append(currPic[1])

    imagesToDisplay.sort(key=lambda x: x[1])

    # Create matplotlib graph and display images
    ax = []
    fig = plt.figure(figsize=(7, 5))
    columns = 5
    rows = 2
    for i in range(1,len(imagesToDisplay)+1):
        img = imagesToDisplay[i-1][0]
        ax.append(fig.add_subplot(rows, columns, i))
        ax[-1].set_title(mydict[imagesToDisplay[i-1][1]])
        plt.imshow(img)
    plt.show()

# Main function to call all image processing functions
def main():
    
    padding_boolean = input("Do you wish to apply padding to the image? (y/n)")
    colored_boolean = input("Is the input image being used colored? (y/n)")
    
    if padding_boolean == "y":
        padding_boolean = True
    else:
        padding_boolean = False

    if colored_boolean == "y":
        colored_boolean = True
    else:
        colored_boolean = False    
    
    if colored_boolean:
        original_image = cv2.imread("sample_image2.png")
    else:
        original_image = cv2.imread("sample_image2.png", 0)
    
    
    downsamplex16 = downsample(original_image, 16, colored=colored_boolean)
    neighborhood_size = 5
    sigma = 5
    alpha = 3 
    beta = 15
    gaussian_kernel = gaussian(neighborhood_size, sigma)
    
    # print("\nNow Processing Top Right Shift...")
    # top_right_shift_image = original_image
    # for i in range(10):
    #     print("\nNow Processing Iteration: " + str(i+1))
    #     top_right_shift_image = topRightShift(top_right_shift_image, padding=padding_boolean, colored=colored_boolean)

    print("\nNow Processing Top Right Shift... (10 Iterations)")
    top_right_shift_image = topRightShift(original_image, colored=colored_boolean, iterations=10)
    
    print("\nNow Processing Gaussian Blur...")
    gaussian_blurred_image = convolve(original_image, gaussian_kernel, padding=padding_boolean, colored=colored_boolean)
    
    print("\nNow Processing Gaussian Blur Difference...")
    gaussian_difference_image = gaussianDifference(original_image, 5, alpha, beta, padding=padding_boolean, colored=colored_boolean)
    
    print("\nNow Processing Sobel X...")
    sobel_X_image = sobelX(original_image, padding_boolean, colored_boolean)
    
    print("\nNow Processing Sobel Y...")
    sobel_Y_image = sobelY(original_image, padding_boolean, colored_boolean)
    
    print("\nNow Processing Orientation Map...")
    orientation_map_image = pixelOrientation(original_image, padding_boolean, colored_boolean)
    
    print("\nNow Processing Magnitude Map...")
    magnitude_map_image = magnitudeMap(original_image, padding_boolean, colored_boolean)
    
    print("\nNow Processing Non Max Suppression...")
    nms_image = nonMaxSuppresion(original_image)

    print("\nNow Processing Canny Edge Detection...")
    canny_image = canny(original_image, 100, 200)
    
    print("\nNow Processing Custom Canny Edge Detection...")
    custom_canny_image = customCanny(original_image, 100, 200, 100)
    
    cv2.imshow("Original Image", original_image)
    cv2.waitKey(0)
    displayImages(original_image, [2,4,8,16], colored=colored_boolean)
    cv2.imshow("Upsampled (Nearest Neighbor)", cv2.resize(downsamplex16, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.imshow("Upsampled (Bilinear Interpolation)", cv2.resize(downsamplex16, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR))
    cv2.waitKey(0)
    cv2.imshow("Upsampled (Bicubic Interpolation)", cv2.resize(downsamplex16, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
    cv2.imshow("Top Right Shift", top_right_shift_image)
    cv2.waitKey(0)
    cv2.imshow("Gaussian Blur", gaussian_blurred_image)
    cv2.waitKey(0)
    cv2.imshow("Gaussian Difference", gaussian_difference_image)
    cv2.waitKey(0)
    cv2.imshow("Sobel X", sobel_X_image)
    cv2.waitKey(0)
    cv2.imshow("Sobel Y", sobel_Y_image)
    cv2.waitKey(0)
    cv2.imshow("Pixel Orientation Map", orientation_map_image)
    cv2.waitKey(0)
    cv2.imshow("Magnitude Map", magnitude_map_image)
    cv2.waitKey(0)
    cv2.imshow("Non Max Suppression", nms_image)
    cv2.waitKey(0)
    cv2.imshow("Canny Edge Detection (OpenCV version)", canny_image)
    cv2.waitKey(0)
    cv2.imshow("Canny Edge Detection (Custom Version)", custom_canny_image)
    cv2.waitKey(0)
    
    displayCIFAR()

if __name__ == "__main__":
    main()