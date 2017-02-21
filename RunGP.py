import GP
import numpy as np
from scipy.misc import imread, imsave
from scipy.ndimage.filters import sobel, laplace, gaussian_laplace
import pickle

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("test_num", type=int,help="set data file")
parser.add_argument("data_set", type=int,help="set data file")
parser.add_argument("data_precision", type=int,help="set data file")

args = parser.parse_args()



print("Reading and generating images...")
gt_img = imread("Images/gt-img1.png", flatten="False", mode="RGB")
images = []
images.append(imread("Images/g-img1.png", flatten="False", mode="RGB"))
#images.append(sobel(images[0], axis=0))
#images.append(sobel(images[0], axis=1))
#mag = np.hypot(images[-1], images[-2])  # magnitude
#mag *= 255.0 / np.max(mag)  # normalize (Q&D)
#imsave('sobel-both.jpg', mag)
#images.append(mag)
images.append(laplace(images[0]))
#imsave('laplace.jpg', images[-1])
#images.append(gaussian_laplace(images[0], 0.05))
images.append(imread("equ_histo_thresh.png", flatten="False", mode="RGB"))
mean_5 = []
mean_11 = []
mean_21 = []
std_5 = []
std_11 = []
std_21 = []
mom_5 = []
mom_11 = []
mom_21 = []
all_mom_5 = []
all_mom_11 = []
all_mom_21 = []

mom_labels = []

#print("Reading in pre-calculated means...")

# img-0 - greyscale
# img-1 - sobel horizontal
# img-2 - sobel vertical
# img-3 - sobel
# img-4 - laplace
# img-5 - gaussian_laplace

for i in range(3):
    print("Reading in precalculated means for image " + str(i))
    f = open("Images/Means/img-"+str(i)+"-5.txt", 'r')
    tmp = []
    for row in f.readlines():
        tmp.append([float(x) for x in row.split()])
    f.close()
    mean_5.append(tmp)

    f = open("Images/Means/img-"+str(i)+"-11.txt", 'r')
    tmp = []
    for row in f.readlines():
        tmp.append([float(x) for x in row.split()])
    f.close()
    mean_11.append(tmp)

    f = open("Images/Means/img-"+str(i)+"-21.txt", 'r')
    tmp = []
    for row in f.readlines():
        tmp.append([float(x) for x in row.split()])
    f.close()
    mean_21.append(tmp)

    print("Reading in precalculated stds for image " + str(i))
    f = open("Images/Deviation/img-"+str(i)+"-5.txt", 'r')
    tmp = []
    for row in f.readlines():
        tmp.append([float(x) for x in row.split()])
    f.close()
    std_5.append(tmp)

    f = open("Images/Deviation/img-"+str(i)+"-11.txt", 'r')
    tmp = []
    for row in f.readlines():
        tmp.append([float(x) for x in row.split()])
    f.close()
    std_11.append(tmp)

    f = open("Images/Deviation/img-"+str(i)+"-21.txt", 'r')
    tmp = []
    for row in f.readlines():
        tmp.append([float(x) for x in row.split()])
    f.close()
    std_21.append(tmp)

    
    print("Reading in precalculated moments for image " + str(i))
    f = open("Images/Moments/img-"+str(0)+"-5.txt", 'r')
    tmp = []
    for row in f.readlines():
        tmp2 = [float(x) for x in row.split()]
        tmp3 = [[tmp2[i*4], tmp2[i*4+1], tmp2[i*4+2], tmp2[i*4+3]] for i in range(len(tmp2)//4)]
        tmp.append(tmp3)
    f.close()
    mom_5.append(tmp)

    f = open("Images/Moments/img-"+str(0)+"-11.txt", 'r')
    tmp = []
    for row in f.readlines():
        tmp2 = [float(x) for x in row.split()]
        tmp3 = [[tmp2[i*4], tmp2[i*4+1], tmp2[i*4+2], tmp2[i*4+3]] for i in range(len(tmp2)//4)]
        tmp.append(tmp3)
    f.close()
    mom_11.append(tmp)

    f = open("Images/Moments/img-"+str(0)+"-21.txt", 'r')
    tmp = []
    for row in f.readlines():
        tmp2 = [float(x) for x in row.split()]
        tmp3 = [[tmp2[i*4], tmp2[i*4+1], tmp2[i*4+2], tmp2[i*4+3]] for i in range(len(tmp2)//4)]
        tmp.append(tmp3)
    f.close()
    mom_21.append(tmp)

    # all moments

    print("Reading in precalculated raw moments for image " + str(i))
    f = open("Images/Moments/img-"+str(0)+"-5-all.txt", 'rb')
    data = pickle.load(f)
    d = []
    for r in range(len(data)):
        col = []
        for c in range(len(data[r])):
            tmp = []
            for key in data[r][c].keys():
                tmp.append(data[r][c][key])
            col.append(tmp)
        d.append(col)
    all_mom_5.append(d)

    f = open("Images/Moments/img-"+str(0)+"-11-all.txt", 'rb')
    data = pickle.load(f)
    d = []
    for r in range(len(data)):
        col = []
        for c in range(len(data[r])):
            tmp = []
            for key in data[r][c].keys():
                tmp.append(data[r][c][key])
            col.append(tmp)
        d.append(col)
    all_mom_11.append(d)

    f = open("Images/Moments/img-"+str(0)+"-21-all.txt", 'rb')
    data = pickle.load(f)
    d = []
    for r in range(len(data)):
        col = []
        for c in range(len(data[r])):
            tmp = []
            for key in data[r][c].keys():
                tmp.append(data[r][c][key])
                if (i == 0 and r == 0 and c == 0):
                    mom_labels.append(key)
            col.append(tmp)
        d.append(col)
    all_mom_21.append(d)


#print ("Mean", len(mean_5[0]), len(mean_5[0][0]))
#print ("Mean", len(mean_5[1]), len(mean_5[1][0]))
#print ("Mean", len(mean_5[2]), len(mean_5[2][0]))


#print ("Mean", len(mean_5[0]), len(mean_5[0][0]))
#print ("Mean", len(mean_5[1]), len(mean_5[1][0]))
#print ("Mean", len(mean_5[2]), len(mean_5[2][0]))

#print ("Moment", len(mom_5[0]), len(mom_5[0][0]))


'''
f = open("Images/Means/img-0-17.txt", 'r')
for row in f.readlines():
    mean_17.append([float(x) for x in row.split()])
f.close()

f = open("Images/Means/img-0-21.txt", 'r')
for row in f.readlines():
    mean_21.append([float(x) for x in row.split()])
f.close()

f = open("Images/Deviation/img-0-13.txt", 'r')
for row in f.readlines():
    std_13.append([float(x) for x in row.split()])
f.close()

f = open("Images/Deviation/img-0-17.txt", 'r')
for row in f.readlines():
    std_17.append([float(x) for x in row.split()])
f.close()

f = open("Images/Deviation/img-0-21.txt", 'r')
for row in f.readlines():
    std_21.append([float(x) for x in row.split()])
f.close()
'''

print("Reading in training points...")
#f = open("img_1_datapoints.txt")
f = open("Rand_datapoints4.txt")
points = []
for line in f.readlines()[1:]:
    points.append([int(l) for l in line.split()])
    #print (x) for x in line.split()[:2])
#print(points)

print("Setting up data for use...")

#labels = ["gt", "grey", "mean13_13", "mean17_17", "mean21_21", "std13_13", "std17_17", "std21_21","sobel_v","sobel_h","laplace","gaussian_laplace"]

s = args.data_set
#i = args.data_precision

data = []
labels = []
print ("Adding Ground Truth and base values")
# 0 raw value from last assignment
# 1 all raw values
# 2 raw and rest from original
# 3 raw and rest from all
# 4 - 3 + raw moments
# 5 - 3 + calculated moments
print ("Adding raw for grey and laplace")
labels.append("img_gt")
labels.append("grey_val")
labels.append("lap_val")
# Add Thresholded image
if (s == 1 or s > 2):
    print ("Adding raw for thresholded")
    labels.append("thresh_val")

if (s > 1):
    #print ("Adding mean and std for grey and laplace")
    #if (s > 2):
    #    print ("Adding mean and std for thresholded")
    for i in ["5", "11", "21"]:
        labels.append("grey_mean_"+i)
        labels.append("grey_std_"+i)
        labels.append("lap_mean_"+i)
        labels.append("lap_std_"+i)
        if (s > 2):
            labels.append("thresh_mean_"+i)
            labels.append("thresh_std_"+i)

if (s == 4):
    #print ("Adding raw moments")
    for i in ["grey", "laplace"]:
        for j in ["5", "11", "21"]:
            for key in mom_labels:
                labels.append(i+"_"+str(key)+j)
    

if (s == 5):
    #print ("Adding calculated moments")
    for i in ["grey", "laplace"]:
        for j in ["5", "11", "21"]:
            labels.append(i+"_cx_"+j)
            labels.append(i+"_cy_"+j)
            labels.append(i+"_length_"+j)
            labels.append(i+"_width_"+j)


for r in range(len(gt_img)):
    row = []
    for c in range(len(gt_img[r])):
        tmp = []

        #print("Adding raw values for laplace and grayscale")
        # Append all image raw pixel values
        tmp.append(gt_img[r][c])
        #for i in range(len(images)):
        tmp.append(images[0][r][c])
        tmp.append(images[1][r][c])
        if (s == 1 or s > 2):
            tmp.append(images[2][r][c])
        #    print("adding raw values for thresholded image")
            

        # Append the Means and stds of each image
        if (s > 1): 
            tmp.append(mean_5[0][r][c])
            tmp.append(std_5[0][r][c])
            tmp.append(mean_5[1][r][c])
            tmp.append(std_5[1][r][c])
#    print("Adding Means and stds for laplace and Grayscale")
            if (s > 2):
                tmp.append(mean_5[2][r][c])
                tmp.append(std_5[2][r][c])
       #        print("Adding means and stds for thresholded image")
            
            tmp.append(mean_11[0][r][c])
            tmp.append(std_11[0][r][c])
            tmp.append(mean_11[1][r][c])
            tmp.append(std_11[1][r][c])
            if (s > 2):
                tmp.append(mean_11[2][r][c])
                tmp.append(std_11[2][r][c])
            
            tmp.append(mean_21[0][r][c])
            tmp.append(std_21[0][r][c])
            tmp.append(mean_21[1][r][c])
            tmp.append(std_21[1][r][c])
            if (s > 2):
                tmp.append(mean_21[2][r][c])
                tmp.append(std_21[2][r][c])
            

        if (s == 4):
            for i in range(len(images)-1):
            #    print("Adding all moments for image " + str(i))
                for a in range(len(all_mom_5[i][r][c])):
                    tmp.append(all_mom_5[i][r][c][a])
                
                for a in range(len(all_mom_11[i][r][c])):
                    tmp.append(all_mom_11[i][r][c][a])
                
                for a in range(len(all_mom_21[i][r][c])):
                    tmp.append(all_mom_21[i][r][c][a])

        if (s == 5):
            for i in range(len(images)-1):
            #    print("Adding mom 5, mom 11, mom 21 for image" + str(i))
                for a in range(len(mom_5[i][r][c])):
                    tmp.append(mom_5[i][r][c][a])
                for a in range(len(mom_11[i][r][c])):
                    tmp.append(mom_11[i][r][c][a])
                for a in range(len(mom_21[i][r][c])):
                    tmp.append(mom_21[i][r][c][a])

        row.append(tmp)
    data.append(row)

    


'''
if (i == 1):
    print("Adding greyscale means")
    labels.append("grey_mean_5")
    labels.append("grey_mean_5")
    labels.append("grey_mean_5")
if (i == 2):
    print("Adding greyscale standard deviations")
    labels.append("grey_std_13")
    labels.append("grey_std_17")
    labels.append("grey_std_21")

if (s == 0 or s == 2 or s == 3):
    print("Adding Basic Sobel")
    labels.append("sobel_val")
    if (i == 1):
        print("Adding Sobel means")
        labels.append("sobel_mean_13")
        labels.append("sobel_mean_17")
        labels.append("sobel_mean_21")
    if (i == 2):
        print("Adding Sobel standard deviation")
        labels.append("sobel_std_13")
        labels.append("sobel_std_17")
        labels.append("sobel_std_21")

if (s == 1 or s == 2 or s == 4):
    print("Adding Basic Laplace")
    labels.append("laplace_val")
    if (i == 1):
        print("Adding Laplace means")
        labels.append("laplace_mean_13")
        labels.append("laplace_mean_17")
        labels.append("laplace_mean_21")
    if (i == 2):
        print("Adding Laplace standard deviation")
        labels.append("laplace_std_13")
        labels.append("laplace_std_17")
        labels.append("laplace_std_21")

if (s == 2 or s == 3 or s == 4):
    print("Adding Basic Gaussian_Laplace")
    labels.append("g_laplace_val")
    if (i == 1):
        print("Adding Gaussian_Laplace means")
        labels.append("g_laplace_mean_13")
        labels.append("g_laplace_mean_17")
        labels.append("laplace_mean_21")
    if (i == 2):
        print("Adding Gaussian_Laplace standard deviations")
        labels.append("g_laplace_std_13")
        labels.append("g_laplace_std_17")
        labels.append("g_laplace_std_21")
'''
'''
for y in range(len(gt_img)):
    row = []
    for x in range(len(gt_img[y])):
        #print(images[0][y][x])
        #tmp = [int(gt_img[y][x]), int(images[0][y][x]), mean_13[y][x], mean_17[y][x], mean_21[y][x], std_13[y][x], std_17[y][x], std_21[y][x],
        #int(images[2][y][x])%255, int(images[1][y][x])%255, int(images[3][y][x])%255, int(images[4][y][x])%255]
        tmp = []
        tmp.append(int(gt_img[y][x]))

        tmp.append(int(images[0][y][x]))

        if (i == 1):
            tmp.append(mean_13[0][y][x])
            tmp.append(mean_17[0][y][x])
            tmp.append(mean_21[0][y][x])
        if (i == 2):
            tmp.append(std_13[0][y][x])
            tmp.append(std_17[0][y][x])
            tmp.append(std_21[0][y][x])

        # img-0 - greyscale
        # img-1 - sobel horizontal
        # img-2 - sobel vertical
        # img-3 - sobel
        # img-4 - laplace
        # img-5 - gaussian_laplace

        # Add sobel information depending on data run
        if (s == 0 or s == 2 or s == 3):
            tmp.append(int(images[3][y][x]))
            if (i == 1):
                tmp.append(mean_13[3][y][x])
                tmp.append(mean_17[3][y][x])
                tmp.append(mean_21[3][y][x])
            if (i == 2):
                tmp.append(std_13[3][y][x])
                tmp.append(std_17[3][y][x])
                tmp.append(std_21[3][y][x])

        if (s == 1 or s == 2 or s == 4):
            tmp.append(int(images[4][y][x]))
            if (i == 1):
                tmp.append(mean_13[4][y][x])
                tmp.append(mean_17[4][y][x])
                tmp.append(mean_21[4][y][x])
            if (i == 2):
                tmp.append(std_13[4][y][x])
                tmp.append(std_17[4][y][x])
                tmp.append(std_21[4][y][x])

        if (s == 2 or s == 3 or s == 4):
            tmp.append(int(images[5][y][x]))
            if (i == 1):
                tmp.append(mean_13[5][y][x])
                tmp.append(mean_17[5][y][x])
                tmp.append(mean_21[5][y][x])
            if (i == 2):
                tmp.append(std_13[5][y][x])
                tmp.append(std_17[5][y][x])
                tmp.append(std_21[5][y][x])

        # sobels:mean_17[y][x],
        #print (tmp)
        row.append(tmp)
    data.append(row)
    '''
# num gens, pop size, cross rate, mut rate
gp = GP.GP(60, 750, 0.85, 0.1)
gp.setData(data, labels)
gp.setTrain(points, gt_img)
gp.setUpGP("Data-" + str(s) + "-Test", args.test_num)
gp.runGP()

f = open("Completed/Complete_Log.txt", 'a+')
f.write("Completed: " + "Data-" + str(s) + "-Test" + str(args.test_num)+"\n")
f.close()
