import cv2
import numpy as np


# Function to convert points from hough space back to parameter space
# and draw lines on to the image

def drawLines(img, lines):
    for line in lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        p1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        p2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        
        cv2.line(img, p1, p2, (0,0,255), 1, cv2.LINE_4)
    
    return img

# Function that creates accumulator hough space and finds maxima's in the hough space
def hough(img, threshold):
    rows, cols = img.shape
    lines = []
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    diag_len = int(np.ceil(np.sqrt(rows ** 2 + cols ** 2)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)
    
    for point in range(len(y_idxs)):
        x = x_idxs[point]
        y = y_idxs[point]
        
        for index, theta in enumerate(thetas):
            
            rho = int(x*np.cos(theta) + y*np.sin(theta)) + diag_len
            accumulator[rho, index] += 2
        
    
    for rI in range(accumulator.shape[0]):
        for tI in range(accumulator.shape[1]):
            if accumulator[rI, tI] >= threshold*2:
                lines.append([rI - diag_len, np.deg2rad(tI-90)])
    
    
    
    return accumulator, lines    

# Main function
def main():
    
    # hough1.png - Threshold = 60
    # hough2.png - Threshold = 100
    
    img = cv2.imread("hough/hough1.png")
    original_image = img.copy()
    edges = cv2.Canny(img, 50, 150)
    
    print("\n Now Processing: Hough Line Detection\n")
    
    accumulator, lines = hough(edges, 60)
    accumulator = accumulator.astype(np.uint8)
    lined_image = drawLines(img, lines)
    
    cv2.imshow("Original Image", original_image)
    cv2.waitKey(0)
    cv2.imshow("Hough Space", accumulator)
    cv2.waitKey(0)
    cv2.imshow("Lined Image", lined_image)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    main()