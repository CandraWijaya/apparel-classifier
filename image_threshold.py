import cv2

def threshold(image, source):
    # blur and grayscale before thresholding
    if source == 'image':
    	image = cv2.imread(image)
    blur = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(src = blur, ksize = (7, 7), sigmaX = 0)

    # perform inverse binary thresholding 
    (t, maskLayer) = cv2.threshold(src = blur, 
        thresh = 0, maxval = 255, 
        type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # make a mask suitable for color images
    mask = cv2.merge(mv = [maskLayer, maskLayer, maskLayer])
    
    # use the mask to select the "interesting" part of the image
    sel = cv2.bitwise_and(src1 = image, src2 = mask)

    return sel
    # cv2.waitKey(delay = 0)