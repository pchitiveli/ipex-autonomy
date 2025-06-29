import ultralytics
import cv2
import numpy as np

model = ultralytics.YOLO("model/rock-seg-1.pt")
model.predict(source="sharpened_vid.mp4", show=True, save=True, conf=0.2, line_width=2, save_crop=False, save_txt=False, show_labels=True, show_conf=True)

# img = cv2.imread("images/train/1_151_0.png", cv2.IMREAD_GRAYSCALE)

# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# # img = cv2.filter2D(img, -1, kernel)
# img = cv2.resize(img, (1280, 1280))

# while True:
#     cv2.imshow("Resized", img)
#     cv2.waitKey(1)

# cv2.destroyAllWindows()