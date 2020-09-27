import cv2
import numpy as np


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    cord1 = np.array(p0)
    cord2 = np.array(p1)
    val = np.linalg.norm(cord1 - cord2)
    return val


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    
    length, width  = image.shape[:2]
    c = [(0,0), (0, length-1), (width-1, 0), (width-1, length-1)]
    
    return c

def find_markers(image, template=None):
    """
    Finds markers defining the bounds where the image should be 
    projected on to
    """
    newimg = image.copy()
    newimg = cv2.filter2D(newimg, -1,  0.05 * np.ones((5,5)))
    newimg = cv2.GaussianBlur(newimg, (3,3), 0)
    img = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
    cr  = cv2.cornerHarris(img, 5,3,0.04)
    cr *= (255.0/cr.max())
    cr[cr<30] = 0
    c = np.array([(i,j) for (i,j) in zip(*np.where(cr>0))], dtype = np.float32)
    ret,label,center  = cv2.kmeans(c,4,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.fliplr(center).astype(np.int32)
    scord = np.argsort(center[:,0])
    lpos = center[scord[:2]]
    rpos = center[scord[2:]]
    topl,botl = [tuple(l) for l in lpos[np.argsort(lpos[:,1])]]
    topr,botr = [tuple(l) for l in rpos[np.argsort(rpos[:,1])]]

    corners = [topl, botl, topr, botr]

    return corners

def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    iterlist = [0,1,3,2,0]
    newmarkers = []
    for i in iterlist:
        newmarkers.append(markers[i])

    img = image.copy()
    for cord1, cord2 in zip(newmarkers[:-1], newmarkers[1:]):
        cv2.line(img, cord1, cord2, (255,0,0), thickness=thickness)
    
    return img

def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """ 
    picA = imageA.copy()
    picB = imageB.copy()

    h, w = picB.shape[:2]
    idy,idx = np.indices((h,w),dtype = np.float32)
    hom_idx = np.array([idx.ravel(), idy.ravel(), np.ones_like(idx).ravel()])
    homgph = np.linalg.inv(homography)
    map_id = homgph.dot(hom_idx)
    mapx, mapy = map_id[:-1]/map_id[-1]
    mapx = mapx.reshape(h,w).astype(np.float32)
    mapy = mapy.reshape(h,w).astype(np.float32)
    dst = cv2.remap(picA,mapx,mapy, cv2.INTER_LINEAR)
    mk= cv2.remap(np.ones(picA.shape), mapx, mapy, cv2.INTER_LINEAR)
    final = picB.copy()

    final[mk==1] = dst[mk==1]
    return final


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    A = []
    for (x,y), (u,v) in zip(src_points, dst_points):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])


    A = np.array(A, dtype=np.float)
    B = np.array(dst_points).ravel()
    af = np.linalg.inv(A).dot(B)
    return np.append(np.array(af).flatten(), 1).reshape(3, 3)

def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)
    print (video)
    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None
