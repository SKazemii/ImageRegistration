from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np

print("[INFO] Reading Images!")
img1 = cv2.imread("images/B.png", cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("images/S.png", cv2.COLOR_BGR2GRAY)


recall = np.zeros([1, 5], dtype=float)
thePrecision = np.zeros([1, 5], dtype=float)


# k = 0
# for deg in [0]:  # , 30, 60, 90]:
# rotMat = cv2.getRotationMatrix2D( (img2_ref.shape[1] // 2, img2_ref.shape[0] // 2), deg, 0.8 )
# img2 = cv2.warpAffine(img2_ref, rotMat, (img2_ref.shape[1], img2_ref.shape[0]))
# img2 = img2_ref
# plt.title("Transformed")
# plt.imshow(img2)
# plt.show()

# img2 = img1[20:80, 55:250]
# plt.title("Transformed")
# plt.imshow(img2), plt.show()

# SIFT  ################################################################
print("[INFO] Starting SIFT!")

# Making a instance of class SIFT
sift = cv2.SIFT_create()

print("[INFO] stage 1: extracting features!")
kp1 = sift.detect(img1, None)
kp2 = sift.detect(img2, None)

# showing keypoints in images
img11 = cv2.drawKeypoints(
    img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
img22 = cv2.drawKeypoints(
    img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# storing the image with its keypoints
cv2.imwrite("images/outputs/sift_keypoints_img1.jpg", img11)
cv2.imwrite("images/outputs/sift_keypoints_img2.jpg", img22)

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

print("[INFO] stage 2: computing description!")
kp1, des1 = sift.compute(img1, kp1)
kp2, des2 = sift.compute(img2, kp2)

print("[INFO] descriotion size of SIFT = {}".format(sift.descriptorSize()))


print("[INFO] stage 3: matching features!")

bf = cv2.BFMatcher()

# Get k best matches
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# Draw first 10 matches.
img3 = cv2.drawMatchesKnn(
    img1,
    kp1,
    img2,
    kp2,
    good[:10],
    None,
    flags=2,
)

# showing matches keypoints
plt.title("SIFT")
plt.imshow(img3), plt.show()

# saving matches keypoints in file
cv2.imwrite("images/outputs/sift_result.jpg", img3)

print("[INFO] size of good features = {}".format(len(good[:10])))

print("[INFO] computing Precision!")
correct_matches = int(input("Please enter the number of correct matches:        "))
incorrect_matches = 10 - correct_matches

coresponding_matches = len(good[:20])

thePrecision[0, 0] = correct_matches / (correct_matches + incorrect_matches)

print("[INFO] SIFT was done...!")

# BRISK  ################################################################
print("[INFO] Starting BRISK!")

# Making a instance of class BRISK
BRISK = cv2.BRISK_create()

print("[INFO] stage 1: extracting features!")
kp1 = BRISK.detect(img1, None)
kp2 = BRISK.detect(img2, None)

# showing keypoints in images
img11 = cv2.drawKeypoints(
    img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
img22 = cv2.drawKeypoints(
    img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# storing the image with its keypoints
cv2.imwrite("images/outputs/BRISK_keypoints_img1.jpg", img11)
cv2.imwrite("images/outputs/BRISK_keypoints_img2.jpg", img22)

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

print("[INFO] stage 2: computing description!")
kp1, des1 = BRISK.compute(img1, kp1)
kp2, des2 = BRISK.compute(img2, kp2)

print("[INFO] descriotion size of BRISK = {}".format(BRISK.descriptorSize()))


print("[INFO] stage 3: matching features!")

# create a BFMatcher instance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

# showing matches keypoints
plt.title("BRISK")
plt.imshow(img3), plt.show()

# saving matches keypoints in file
cv2.imwrite("images/outputs/BRISK_result.jpg", img3)

print("[INFO] size of features = {}".format(len(matches[:10])))

print("[INFO] computing Precision!")
correct_matches = int(input("Please enter the number of correct matches:        "))
incorrect_matches = 10 - correct_matches

coresponding_matches = len(matches[:10])

thePrecision[0, 1] = correct_matches / (correct_matches + incorrect_matches)
print("[INFO] BRISK was done...!")

# BRIEF ################################################################
print("[INFO] Starting BRIEF!")

# Making a instance of class FAST for feature detecting
FAST = cv2.FastFeatureDetector_create()
FAST.setThreshold(80)

print("[INFO] stage 1: extracting features!")
kp1 = FAST.detect(img1, None)
kp2 = FAST.detect(img2, None)

# showing keypoints in images
img11 = cv2.drawKeypoints(
    img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
img22 = cv2.drawKeypoints(
    img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# storing the image with its keypoints
cv2.imwrite("images/outputs/BRIEF_keypoints_img1.jpg", img11)
cv2.imwrite("images/outputs/BRIEF_keypoints_img2.jpg", img22)

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

print("[INFO] stage 2: computing description!")

# Making a instance of class BriefDescriptor for descriptor
BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()

kp1, des1 = BRIEF.compute(img1, kp1)
kp2, des2 = BRIEF.compute(img2, kp2)

print("[INFO] descriotion size of BRIEF = {}".format(BRIEF.descriptorSize()))


print("[INFO] stage 3: matching features!")

# create a BFMatcher instance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

# showing matches keypoints
plt.title("BRIEF")
plt.imshow(img3), plt.show()

# saving matches keypoints in file
cv2.imwrite("images/outputs/BRIEF_result.jpg", img3)

print("[INFO] size of features = {}".format(len(matches[:10])))

print("[INFO] computing Precision!")
correct_matches = int(input("Please enter the number of correct matches:        "))
incorrect_matches = 10 - correct_matches

coresponding_matches = len(matches[:10])

thePrecision[0, 2] = correct_matches / (correct_matches + incorrect_matches)
print("[INFO] BRIEF was done...!")

# FREAK ################################################################
print("[INFO] Starting FREAK!")

# Making a instance of class FAST for feature detecting
FAST = cv2.FastFeatureDetector_create()
FAST.setThreshold(80)

print("[INFO] stage 1: extracting features!")
kp1 = FAST.detect(img1, None)
kp2 = FAST.detect(img2, None)

# showing keypoints in images
img11 = cv2.drawKeypoints(
    img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
img22 = cv2.drawKeypoints(
    img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# storing the image with its keypoints
cv2.imwrite("images/outputs/FREAK_keypoints_img1.jpg", img11)
cv2.imwrite("images/outputs/FREAK_keypoints_img2.jpg", img22)

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

print("[INFO] stage 2: computing description!")

# Making a instance of class FREAK for descriptor
FREAK = cv2.xfeatures2d.FREAK_create()

kp1, des1 = FREAK.compute(img1, kp1)
kp2, des2 = FREAK.compute(img2, kp2)

print("[INFO] descriotion size of FREAK = {}".format(FREAK.descriptorSize()))

print("[INFO] stage 3: matching features!")

# create a BFMatcher instance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

# showing matches keypoints
plt.title("FREAK")
plt.imshow(img3), plt.show()

# saving matches keypoints in file
cv2.imwrite("images/outputs/FREAK_result.jpg", img3)

print("[INFO] size of features = {}".format(len(matches[:10])))

print("[INFO] computing Precision!")
correct_matches = int(input("Please enter the number of correct matches:        "))
incorrect_matches = 10 - correct_matches

coresponding_matches = len(matches[:10])

thePrecision[0, 3] = correct_matches / (correct_matches + incorrect_matches)
print("[INFO] FREAK was done...!")

# ORB ################################################################
print("[INFO] Starting ORB!")

# Making a instance of class ORB
ORB = cv2.ORB_create()

print("[INFO] stage 1: extracting features!")
kp1 = ORB.detect(img1)
kp2 = ORB.detect(img2)

print("[INFO] flag done!")

# showing keypoints in images
img11 = cv2.drawKeypoints(
    img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
img22 = cv2.drawKeypoints(
    img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# storing the image with its keypoints
cv2.imwrite("images/outputs/ORB_keypoints_img1.jpg", img11)
cv2.imwrite("images/outputs/ORB_keypoints_img2.jpg", img22)

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

print("[INFO] stage 2: computing description!")
kp1, des1 = ORB.compute(img1, kp1)
kp2, des2 = ORB.compute(img2, kp2)

print("[INFO] descriotion size of ORB = {}".format(ORB.descriptorSize()))


print("[INFO] stage 3: matching features!")

# create a BFMatcher instance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

# showing matches keypoints
plt.title("ORB")
plt.imshow(img3), plt.show()

# saving matches keypoints in file
cv2.imwrite("images/outputs/ORB_result.jpg", img3)

print("[INFO] size of features = {}".format(len(matches[:10])))

print("[INFO] computing Precision!")
correct_matches = int(input("Please enter the number of correct matches:        "))
incorrect_matches = 10 - correct_matches
coresponding_matches = len(matches[:10])

thePrecision[0, 4] = correct_matches / (correct_matches + incorrect_matches)

print("[INFO] ORB was done...!")

# saving Precision in text file
print(thePrecision)
np.savetxt("results.txt", thePrecision)
