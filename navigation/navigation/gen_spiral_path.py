import numpy as np
import cv2
import colorsys
from PIL import Image, ImageDraw
import math

class PredefinedPath():
    def __init__(self, size):
        self.path = []

        self.cell_resolution = 5
        self.size = self.cell_resolution * size
        self.index = 0
        self.amount = self.size
        self.num_cells = self.amount ** 2
        self.d = 0
        self.num = 1
        self.img = np.zeros((self.size, self.size, 3))
        self.power = 0
        self.hue_incr = 10
        self.hue = 0
        self.path_width = 45
        self.keypoints = []
        self.spacing = 60

    def is_empty(self):
        return self.path == []
    
    def get_path_length(self):
        return len(self.path)

    def remove_path(self, n = 1):
        self.keypoints = self.keypoints[n:]

    def display_image(self):
        img = Image.new('RGB', (self.size, self.size))
        for i in range(len(self.keypoints)):
            # make a line from the startPos to the endPos
            draw = ImageDraw.Draw(img)
            draw.point((self.keypoints[i][1], self.keypoints[i][0]), fill=(255, 255, 255))
        img.show()
        

    def get_path(self, n = 5):
        return self.keypoints[:n]

    def generate_path(self, orig_pos):
        pos = [self.path_width, self.path_width]

        self.path.append(pos)
        self.amount -= 2 * self.path_width

        while self.amount > 0:
            next_d = 0
            add = [0,0]
            if self.d == 0:
                next_d = 1
                add = [1,0]
            elif self.d == 1:
                next_d = 2
                add = [0, 1]
            elif self.d == 2:
                next_d = 3  
                add = [-1, 0]
            else:
                next_d = 0
                add = [0, -1]
            cur_pos = [0, 0]
            for i in range(self.amount):
                cur_pos = [pos[0] + i * add[0], pos[1] + i * add[1]]
                # print(cur_pos)
                self.path.append(cur_pos)
                self.img[cur_pos[0], cur_pos[1]] = np.array(colorsys.hsv_to_rgb(self.hue / 360, 1, 1))
                self.index += 1
                self.hue = np.mod((self.hue + self.hue_incr), 360)
            self.power += 1
            self.num += 1
            if self.num == 2:
                self.amount -= 1 * self.path_width
                self.num = 0
            self.d = next_d
            pos = cur_pos

        self.keypoints = self.get_evenly_spaced_points(orig_pos, self.path[0])
        self.keypoints += self.path
        self.keypoints = [(self.keypoints[i][0], self.keypoints[i][1]) for i in range(0, len(self.keypoints), self.cell_resolution)]


    def get_evenly_spaced_points(self, p1, p2):
        points = []
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist == 0:
            return points
        for j in range(int(dist)):
            x = int(x1 + j * dx / dist)
            y = int(y1 + j * dy / dist)
            print()
            points.append((x, y))
        return points

if __name__ == "__main__":
    path = PredefinedPath(200)
    path.generate_path((450, 400))
    path.display_image()