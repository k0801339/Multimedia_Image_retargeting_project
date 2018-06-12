import cv2
import numpy as np

import sys
import os
import logging
from datetime import datetime

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%I:%M:%S')

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0, 0)

def x_gradient(img):
    '''
    Sobel filter: used for computing the gradient, more resistant to noise
    (img, img's depth, compute dx?, computer dy?, kernel size, ...)
    '''
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)


def y_gradient(img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3,
                     scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

# computer energy function
def energy(img):
    blurred = gaussian_blur(img)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    dx = x_gradient(gray)
    dy = y_gradient(gray)
    # final energy = x_grad + y_grad
    return cv2.add(np.absolute(dx), np.absolute(dy))

def calc_optimum_dynamics(energies):
    h, w = energies.shape[:2]
    dynamics = np.zeros((h, w)) # dynamics's shape [502 ,750]

    for i in range(0, w): # 0, 1, ..., w-1
        dynamics[h - 1, i] = energies[h - 1, i]
    
    for curr_row in range(h -2, -1, -1): # h-2, h-1, ..., 0
        for curr_col in range(0, w): # 0, 1, ..., w-1
            curr_min = dynamics[curr_row + 1, curr_col] + energies[curr_row + 1, curr_col]

            for delta in range(-1, 3, 2): # -1, 1
                if delta + curr_col < w and delta + curr_col >= 0:
                    val_to_exam = dynamics[curr_row + 1, curr_col + delta] + energies[curr_row, curr_col]
                    if (curr_min > val_to_exam) :
                        curr_min = val_to_exam
            dynamics[curr_row, curr_col] = curr_min
    return dynamics

def get_w(optimum_energies, seam_energies, i, j, row):
    return seam_energies[row, i] * optimum_energies[row + 1, j]

def calc_matches(energies, optimum_energies, seam_energies, row):
    matches = np.zeros(energies.shape[1])
    matches[0] = get_w(optimum_energies, seam_energies, 0, 0, row)
    matches[1] = max(matches[0] + get_w(optimum_energies, seam_energies, 1, 1, row), get_w(optimum_energies, seam_energies, 0, 1, row) + get_w(optimum_energies, seam_energies, 1, 0, row))

    for col in range(2, energies.shape[1]):
        w1 = matches[col - 1] + get_w(optimum_energies, seam_energies, col, col, row)
        w2 = matches[col - 2] + get_w(optimum_energies, seam_energies, col, col - 1, row) + get_w(optimum_energies, seam_energies, col - 1, col, row)
        matches[col] = max(w1, w2)
    return matches

def increase_seams(energies, optimum_energies, matches, seam_energies, seams, row):
    x = energies.shape[1] - 1
    while x >= 0:
        last_match = 0
        if x != 0:
            last_match = matches[x - 1]

        if matches[x] == last_match + get_w(optimum_energies, seam_energies, x, x, row):
            seams[x].append(x)
            seam_energies[row + 1, x] = seam_energies[row, x] + energies[row + 1, x]
            x = x - 1
        else:
            seams[x - 1].append(x)
            seams[x].append(x - 1)
            
            temp = seams[x]
            seams[x] = seams[x - 1]
            seams[x - 1] = temp

            seam_energies[row + 1, x - 1] = seam_energies[row, x] + energies[row + 1, x - 1]
            seam_energies[row + 1, x] = seam_energies[row, x - 1] + energies[row + 1, x]

            x = x - 2

    return  seam_energies, seams

def get_seams(img, delta):
    print("start resizing")

    # Get the energy of each pixel
    energies = energy(img) # energies's shape [502, 750]

    # Create and compute the DP form
    optimum_energies = calc_optimum_dynamics(energies) # optimum_energies's shape [502, 750]

    # print("optimum_energies: shape[" , optimum_energies.shape[:2], "]\n", optimum_energies)

    # Initial the seams list
    seams = [] # seams's shape [750, 502]
    for i in range(0, energies.shape[1]):
        seams.append([])
    
    seam_energies = np.zeros((img.shape[0], img.shape[1])) # seams_energies's shape [502, 750]

    for i in range(0, energies.shape[1]):
        seam_energies[0, i] = energies[0, i]
        seams[i].append(i)

    # print("seam_energies\n", seam_energies)
    # print("seams: shape[", np.shape(seams), "]\n", seams)
    
    # Get seams by making bipartite matching
    for row in range(0, energies.shape[0] - 1):
        matches = calc_matches(energies, optimum_energies, seam_energies, row)
        seam_energies, seams = increase_seams(energies, optimum_energies, matches, seam_energies, seams, row)
    
    # print("seam_energies\n", seam_energies)
    # print("seams: shape[", np.shape(seams), "]\n", seams[0])
    weighted_seams = []

    # To sort the seams by its weight
    for i in range(0, energies.shape[1]):
        weighted_seams.append((seam_energies[energies.shape[0] - 1, i], seams[i]))

    # for i in range(0, energies.shape[1]):
    #         print("weighted_seams\n", weighted_seams[i][0])

    weighted_seams = sorted(weighted_seams, key=lambda weighted_seams: weighted_seams[0])

    # for i in range(0, energies.shape[1]):
    #     print("weighted_seams\n", weighted_seams[i][0])

    res = []

    for i in range(0, delta):
        res.append(weighted_seams[i][1])

    return res

def process_seams(img, seams, delete_mode):
    print("Process image...")
    if delete_mode:
        w_delta = np.shape(seams)[0] * -1
    else:
        w_delta = np.shape(seams)[0] * 1
    print("w_delta ", w_delta)
    out = np.zeros((np.shape(img)[0], np.shape(img)[1] + w_delta, 3), np.uint8)

    for row in range(0, np.shape(img)[0]):
        pool = []
        for seam in seams:
            pool.append(seam[np.shape(img)[0] - row - 1])

        pool.sort()

        delta = 0
        curr_pix = 0
        for col in range(0, np.shape(img)[1]):
            if curr_pix < np.shape(pool)[0] and col == pool[curr_pix]:
                curr_pix = curr_pix + 1
                if delete_mode:
                    delta = delta - 1
                else:
                    delta = delta + 1

                if delete_mode:
                    continue

                out[row, col + delta - 1, :] = img[row, col, :]
            
            out[row, col + delta, :] = img[row, col, :]
    
    print("Process image done")
    return out

def _resize_width(img, delta):
    if delta == 0:
        print("This side's delta is 0, so do nothing")
        return img
    # Start to resizing
    seams = get_seams(img, abs(delta))
    return process_seams(img, seams, delta < 0)

def resize(img, width=None, height=None):
    result = np.copy(img) # img's shape [502, 750]

    # Get original image width and height
    origin_img_height, origin_img_width = img.shape[:2]

    # Compute the scaling of width and height
    w_delta = width - origin_img_width
    h_delta = height - origin_img_height
    if w_delta < 0:
        w_delta = min(abs(w_delta), origin_img_width) * -1
    else:
        w_delta = min(abs(w_delta), origin_img_width) * 1
       
    if h_delta < 0:
        h_delta = min(abs(h_delta), origin_img_height) * -1
    else:
        h_delta = min(abs(h_delta), origin_img_height) * 1

    print("w_delta: " + str(w_delta) + ", h_delta: " + str(h_delta))

    # We do not transpose the result to make one orientaiton (row or col) calculation 

    print("From (" + str(origin_img_height)  + "," + str(origin_img_width) + ")" + " resize to (" + str(height) + "," + str(width) + ")")

    # Start to resizing
    print("Resizing one side")

    result = _resize_width(result, w_delta)

    print("Resizing another side")
   
    h, w, d = np.shape(result)
    temp_result = np.zeros((w, h, d), np.uint8)
    temp_result[:, :, 0] = result[:, :, 0].transpose()
    temp_result[:, :, 1] = result[:, :, 1].transpose()
    temp_result[:, :, 2] = result[:, :, 2].transpose()

    print(np.shape(result), " transpose to ", np.shape(temp_result))

    temp_result = _resize_width(temp_result, h_delta)

    h, w, d = np.shape(temp_result)
    result = np.zeros((w, h, d), np.uint8)
    result[:, :, 0] = temp_result[:, :, 0].transpose()
    result[:, :, 1] = temp_result[:, :, 1].transpose()
    result[:, :, 2] = temp_result[:, :, 2].transpose()

    # give output image name
    name = sys.argv[1][:-3] + '_content_aware_resize.jpg'
    cv2.imwrite(name, result)
    # show the computing time
    logging.info("Processing took %f sec" % ((datetime.now() - start_time).total_seconds()))
    
    print ('Press any key to close the window.')

    cv2.imshow('seam', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

def usage(program_name):
    '''Usage: python {} image [--interactive] [new_width new_height]

    --interactive        After starting the program, click in the window to pick
                         the height and width to resize to. Once you've made
                         your final selection, press any key to start the seam
                         carving process.
    new_width            The width to resize the image to.
    new_height           The height to resize the image to.
    '''
    #print (''.format(program_name))

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    logging.info("Processing image")
    start_time = datetime.now()
    if len(sys.argv) == 4:
        resize(img, int(sys.argv[2]), int(sys.argv[3]))
    else:
        usage(sys.argv[0])
        sys.exit(1)
    