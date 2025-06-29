from enum import Enum
from PIL import Image, ImageDraw

class line:
    def __init__(self, pos, end):
        self.startPos = pos
        self.endPos = end
        self.purpose = None


states = Enum('states', [('INIT', 0), ('OUT', 1), ('RETURN', 2), ('TRAVEL', 3)])

def visualizePath(path):
    img = Image.new('RGB', (2700, 2700))
    for i in range(len(path)):
        # make a line from the startPos to the endPos
        draw = ImageDraw.Draw(img)
        draw.line((path[i].startPos, path[i].endPos), fill=(255, 255, 255), width=3)
    img.show()

if __name__ == '__main__':
    path = []
    path.append(line((0, 0), (100, 100)))
    path.append(line((100, 100), (200, 500)))
    path.append(line((200, 500), (300, 300)))
    visualizePath(path)
