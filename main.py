from cv2 import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("1.jpeg", cv2.COLOR_BGR2GRAY)  # queryImage
# img2 = cv2.imread("2.jpeg", cv2.COLOR_BGR2GRAY)  # trainImage

# rotMat = cv2.getRotationMatrix2D((90, 125), 45, 1.0)
# img2 = cv2.warpAffine(img1, rotMat, img1.shape[:2])
# plt.title("Transformed")
# plt.imshow(img2), plt.show()


img2 = img1[20:80, 55:250]
plt.title("Transformed")
plt.imshow(img2), plt.show()


# SIFT  ################################################################
print("Starting SIFT!")

sift = cv2.SIFT_create()

kp1 = sift.detect(img1, None)
kp2 = sift.detect(img2, None)

# img1 = cv2.drawKeypoints(
#     img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
# img2 = cv2.drawKeypoints(
#     img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )

cv2.imwrite("sift_keypoints_img1.jpg", img1)
cv2.imwrite("sift_keypoints_img2.jpg", img2)

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

print(kp1)

kp1, des1 = sift.compute(img1, kp1)
kp2, des2 = sift.compute(img2, kp2)

print(kp1)
print(sift.descriptorSize())


bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(
    img1,
    kp1,
    img2,
    kp2,
    good,
    None,
    flags=2,
)

plt.title("SIFT")
plt.imshow(img3), plt.show()

cv2.imwrite("sift_result.jpg", img3)

print("sift was done...!")


# BRISK  ################################################################
print("Starting BRISK!")
BRISK = cv2.BRISK_create()

kp1 = BRISK.detect(img1, None)
kp2 = BRISK.detect(img2, None)

# img1 = cv2.drawKeypoints(
#     img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
# img2 = cv2.drawKeypoints(
#     img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )

cv2.imwrite("BRISK_keypoints_img1.jpg", img1)
cv2.imwrite("BRISK_keypoints_img2.jpg", img2)

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

print(kp1)

kp1, des1 = BRISK.compute(img1, kp1)
kp2, des2 = BRISK.compute(img2, kp2)

print(kp1)
print(BRISK.descriptorSize())


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)


plt.title("BRISK")
plt.imshow(img3), plt.show()

cv2.imwrite("BRISK_result.jpg", img3)

print("BRISK was done...!")


# BRIEF ################################################################
print("Starting BRIEF!")

FAST = cv2.FastFeatureDetector_create()
FAST.setThreshold(80)

kp1 = FAST.detect(img1, None)
kp2 = FAST.detect(img2, None)

# img1 = cv2.drawKeypoints(
#     img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
# img2 = cv2.drawKeypoints(
#     img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )

cv2.imwrite("BRIEF_keypoints_img1.jpg", img1)
cv2.imwrite("BRIEF_keypoints_img2.jpg", img2)

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

print(kp1)

BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()

kp1, des1 = BRIEF.compute(img1, kp1)
kp2, des2 = BRIEF.compute(img2, kp2)

print(kp1)
print(BRIEF.descriptorSize())


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)


plt.title("BRIEF")
plt.imshow(img3), plt.show()

cv2.imwrite("BRIEF_result.jpg", img3)

print("BRIEF was done...!")


# FREAK ################################################################
print("Starting FREAK!")
FAST = cv2.FastFeatureDetector_create()
FAST.setThreshold(80)


kp1 = FAST.detect(img1, None)
kp2 = FAST.detect(img2, None)


# img1 = cv2.drawKeypoints(
#     img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
# img2 = cv2.drawKeypoints(
#     img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )

cv2.imwrite("FREAK_keypoints_img1.jpg", img1)
cv2.imwrite("FREAK_keypoints_img2.jpg", img2)

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

print(kp1)

FREAK = cv2.xfeatures2d.FREAK_create()

kp1, des1 = FREAK.compute(img1, kp1)
kp2, des2 = FREAK.compute(img2, kp2)

print(kp1)
print(FREAK.descriptorSize())


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)


plt.title("FREAK")
plt.imshow(img3), plt.show()

cv2.imwrite("FREAK_result.jpg", img3)

print("FREAK was done...!")

# ORB ################################################################
print("Starting ORB!")
ORB = cv2.ORB_create()

kp1 = ORB.detect(img1)
kp2 = ORB.detect(img2)

print("flag done!")


# img1 = cv2.drawKeypoints(
#     img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
# img2 = cv2.drawKeypoints(
#     img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )

cv2.imwrite("ORB_keypoints_img1.jpg", img1)
cv2.imwrite("ORB_keypoints_img2.jpg", img2)

# plt.imshow(img1), plt.show()
# plt.imshow(img2), plt.show()

print(kp1)

kp1, des1 = ORB.compute(img1, kp1)
kp2, des2 = ORB.compute(img2, kp2)

print(kp1)
print(ORB.descriptorSize())


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)


plt.title("ORB")
plt.imshow(img3), plt.show()

cv2.imwrite("ORB_result.jpg", img3)

print("ORB was done...!")