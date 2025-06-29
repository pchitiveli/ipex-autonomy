import numpy as np

INITIAL_WIDTH = 1280
INITIAL_HEIGHT = 720
PRINCIPAL_POINT = (INITIAL_WIDTH / 2.0, INITIAL_HEIGHT / 2.0)
FIELD_OF_VIEW = 70  # degrees
FOCAL_LENGTH_X = INITIAL_WIDTH / (
    2 * np.tan(np.radians(FIELD_OF_VIEW / 2))
)  # calculate focal length
FOCAL_LENGTH_Y = INITIAL_HEIGHT / (
    2 * np.tan(np.radians(39.375 / 2))
)  # calculate focal length

print(FOCAL_LENGTH_X, FOCAL_LENGTH_Y)

BASELINE = 0.162
print(BASELINE * FOCAL_LENGTH_X)