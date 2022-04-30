import cv2
import numpy as np
from feature import Feature


# Return sobel X 
def gradient_x(img):
    grad_img = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    return grad_img/np.max(grad_img)

# Return sobel Y
def gradient_y(img):
    grad_img = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    return grad_img/np.max(grad_img)

# Harris Corner Detector
def harris(img, threshold, threshold_NMS):
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur image and get gradients
    blur_img = cv2.GaussianBlur(img, (3,3), 1)
    Ix = gradient_x(blur_img)
    Iy = gradient_y(blur_img)

    # Get squared gradients
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    x_y_R = []
    k = 0.04
    max = 0

    # Iterate over image with windows from squared gradients
    for i in range(1, int(img.shape[0] - 1)) :
            for j in range(1, int(img.shape[1] - 1)) :
                
                window_x = Ixx[i-4 : i+5 , j-4 : j+5]
                window_y = Iyy[i-4 : i+5 , j-4 : j+5]
                window_xy = Ixy[i-4 : i+5 , j-4 : j+5] 
                
                # Calculate determinant and trace to check if point is a corner
                determinant = (np.sum(window_x) * np.sum(window_y)) - (np.sum(window_xy)**2)
                trace = np.sum(window_x) + np.sum(window_y)
                R = determinant - (k * trace**2)
                x_y_R.append((i, j, R))
                if(R > max) :
                    max = R

    # Threshold points
    thresholded_points = []

    for res in x_y_R:
        i, j, R = res
        if R > threshold:
            thresholded_points.append([i, j, R])
        

    sorted_thresholded_points = sorted(thresholded_points, key = lambda x: x[2], reverse = True)
    final_points = []
    final_points.append(sorted_thresholded_points[0][:-1])

    # Apply non maximal suppression
    xc, yc = [], []
    for i in sorted_thresholded_points :
        for j in final_points :
            if(abs(i[0] - j[0] <= threshold_NMS) and abs(i[1] - j[1]) <= threshold_NMS) :
                break
        else :
            final_points.append(i[:-1])
            xc.append(i[1])
            yc.append(i[0])


    # Create corner image and return list of feature objects
    corner_img = np.zeros(img.shape)
    listOfFeatureObjects = []
    for i in final_points :
        y, x = i[0], i[1]
        corner_img[y,x] = 1
        temp = cv2.KeyPoint(x, y, 2)
        listOfFeatureObjects.append(Feature(temp, np.array([]).astype('float32'), 0))
        
    return(Ix, Iy, Ixy, listOfFeatureObjects, corner_img)

# Calculate dominant orientation of a feature
def generateFeatureOrientation(listOfFeatureObjects, Ix, Iy):
    
    # Initialize histogram
    total_histogram_values = np.histogram(-1, bins=np.linspace(0, 2*np.pi, 36))[0]
    
    # Iterate over features
    for feature in listOfFeatureObjects:
        
        # Use orientation map and get angle at a point
        orientation_map = np.arctan2(Iy, Ix)
        if len(orientation_map.shape) == 3:
            orientation_map = cv2.cvtColor(orientation_map, cv2.COLOR_BGR2GRAY)
        
        y_location = int(feature.getY())
        x_location = int(feature.getX())
        
        histogram_36bin_values, histogram_36bin_keys  = np.histogram(-1, bins=np.linspace(0, 2*np.pi, 36))
        
        y_start = y_location - 8
        x_start = x_location - 8
        y_end = y_location + 8
        x_end = x_location + 8
        
        # 16 by 16 neighborhood around point
        neighborhood = orientation_map[y_start:y_end, x_start:x_end]

        # Iterate over neighborhood and add to histogram
        for y in range(neighborhood.shape[0]):
            
            for x in range(neighborhood.shape[1]):
                
                if neighborhood[y, x] < 0:
                    histogram_36bin_values, _  = np.histogram(neighborhood[y,x]+(2*np.pi), bins=np.linspace(0, 2*np.pi, 36))
                else:
                    histogram_36bin_values, _  = np.histogram(neighborhood[y,x], bins=np.linspace(0, 2*np.pi, 36))
                
                total_histogram_values += histogram_36bin_values
        
                
        max_occuring_orientation = sorted(list(zip(histogram_36bin_keys, total_histogram_values)), key=lambda n: n[1])[-1]
        
        # Assign max occuring angle to the feature's orientation
        feature.orientation = max_occuring_orientation[0]
    
    return listOfFeatureObjects



# Calculate 128 vector descriptor                                      
def generateDescriptors(listOfFeatureObjects, Ix, Iy):
    
    # Iterate over each feature
    for feature in listOfFeatureObjects:
        
        # Initialize histogram
        total_histogram_values = (np.histogram(-1, bins=np.linspace(0, 2*np.pi, 9))[0]).astype('float32')

        # Calculate orientation map
        orientation_map = np.arctan2(Iy, Ix)
        curr_descriptor = []
        
        y_location = int(feature.getY())
        x_location = int(feature.getX())
        
        
        y_start = y_location - 8
        x_start = x_location - 8
        y_end = y_location + 8
        x_end = x_location + 8
        
        # Get 16 by 16 neighborhood around a point
        neighborhood = orientation_map[y_start:y_end, x_start:x_end]

        # Split 16 by 16 neighborhood into 4x4 chunks
        if neighborhood.shape[0] == 16 and neighborhood.shape[1] == 16:
            chunks = []
            rows_of_4 = np.split(neighborhood, 4)
            for block_of_row in rows_of_4:
                curr_chunks = np.split(block_of_row, 4, axis=1)
                chunks += curr_chunks
                
        # For each chunk get the angle of each point wrt the dominant feature angle
        for chunk in chunks:
            
            total_histogram_values = np.histogram(-1, bins=np.linspace(0, 2*np.pi, 9))[0]

            for y in range(0, 4):
                
                for x in range(0, 4):
                    
                    if len(chunk) != 0:
                        angle = feature.getOrientation()-chunk[y,x]
                        if angle < 0:
                            angle += 2*np.pi
                        histogram_8bin_values = np.histogram(angle, bins=np.linspace(0, 2*np.pi, 9))[0]
                        total_histogram_values += histogram_8bin_values
                        total_histogram_values = total_histogram_values.astype('float32')
            curr_descriptor += list(total_histogram_values)
        
        # Add to feature descriptor
        feature.descriptor = np.array(curr_descriptor)
    
    return listOfFeatureObjects

            
# Get sum of square differences for two features
def SSD(featureObject1, featureObject2):
    toReturn = 0
    for i in range(len(featureObject1.getDescriptor())):
        toReturn += (featureObject1.getDescriptor()[i] - featureObject2.getDescriptor()[i])**2
    return toReturn

# SSD Ratio test to find best matches
def generateMatches(listOfFeatureObjects1, listOfFeatureObjects2, threshold=0.8):
    
    matches = []
    
    # Iterate over all features
    for i in range(0, len(listOfFeatureObjects1)):
        
        # Initialize values
        best_match_index = 0
        best_match = listOfFeatureObjects2[0]
        best_match_SSD = SSD(listOfFeatureObjects1[i], listOfFeatureObjects2[0])
        second_best_SSD = SSD(listOfFeatureObjects1[i], listOfFeatureObjects2[0])
        
        # Iterate over all features in second list to find best match
        for j in range(0, len(listOfFeatureObjects2)):
            
            # Calculate SSD
            curr_ssd = SSD(listOfFeatureObjects1[i], listOfFeatureObjects2[j])
            
            # If current SSD is better than best found SSD
            if (curr_ssd < best_match_SSD):
                best_match_SSD = curr_ssd
                best_match = listOfFeatureObjects2[j]
                best_match_index  = j
        
        # Iterate over all features in second list to find second best match
        for j in range(0, len(listOfFeatureObjects2)):
            curr_ssd = SSD(listOfFeatureObjects1[i], listOfFeatureObjects2[j])
            
            # If current ssd is better than second best and not equal to the best found match
            if (curr_ssd < second_best_SSD and not equal(listOfFeatureObjects2[j], best_match)):
                second_best_SSD = curr_ssd
        if (best_match_SSD == second_best_SSD or best_match_SSD/second_best_SSD > threshold):
            continue
        else:
            matches.append(cv2.DMatch(i, best_match_index, 0))

    return matches

# Check if two feature objects are equal
def equal(featureObject1, featureObject2):
    return(featureObject1.equal(featureObject2))



def main():
    
    # Threshold = 4, NMS = 16 (T Image)
    # Threshold = 10, NMS = 20 (Contrast Image)

    imgT = cv2.imread("hough/hough1.png")
    img = cv2.imread("image_sets/yosemite/Yosemite1.jpg")
    img2 = cv2.imread("image_sets/yosemite/Yosemite2.jpg")
    
    temp = []
    
    print("\n Now Processing: Harris Corner Detection\n")
    Ix, Iy, Ixy, features, corner_img = harris(imgT, 4, 16)
    
    
    
    cv2.imshow("Ix", Ix)
    cv2.waitKey(0)
    cv2.imshow("Iy", Iy)
    cv2.waitKey(0)
    cv2.imshow("Ixy", Ixy)
    cv2.waitKey(0)
    cv2.imshow("Corner Response", corner_img)
    cv2.waitKey(0)
    for feature in features:
        temp.append(feature.getFeature())
    imgT = cv2.drawKeypoints(imgT, temp, None, (0,0,255))
    cv2.imshow("Detected Corners", imgT)
    cv2.waitKey(0)
    
    Ix, Iy, Ixy, listOfFeatureObjects1, corner_img = harris(img, 10, 20)
    Ix2, Iy2, Ixy2, listOfFeatureObjects2, corner_img2 = harris(img2, 10, 20)
    
    print("\n Now Processing: SIFT Descriptors\n")

    listOfFeatureObjects1 = generateDescriptors(generateFeatureOrientation(listOfFeatureObjects1, Ix, Iy), Ix, Iy)
    listOfFeatureObjects2 = generateDescriptors(generateFeatureOrientation(listOfFeatureObjects2, Ix2, Iy2), Ix2, Iy2)
    
    kp1 = []
    kp2 = []
    
    for k in listOfFeatureObjects1:
        kp1.append(k.getFeature())
        
    for k in listOfFeatureObjects2:
        kp2.append(k.getFeature())
    
    matches = generateMatches(listOfFeatureObjects1, listOfFeatureObjects2)

    img3 = cv2.drawMatches(img, kp1, img2, kp2, matches, None)
    
    cv2.imshow("Custom Matches", img3)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()