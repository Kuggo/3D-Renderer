import math
import pygame.draw
from pygame import display, event, QUIT, time


# 2D Screen objects

class Color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
        return

    @staticmethod
    def from_tuple(t: tuple[int, int, int]):
        return Color(t[0], t[1], t[2])

    def tuple(self):
        return self.r, self.g, self.b

    def round(self):
        return Color(round(self.r), round(self.g), round(self.b))

    def __add__(self, other: 'Color'):
        return Color(self.r + other.r, self.g + other.g, self.b + other.b)

    def __sub__(self, other: 'Color'):
        return Color(self.r - other.r, self.g - other.g, self.b - other.b)

    def __mul__(self, other: int|float):
        return Color(self.r * other, self.g * other, self.b * other)

    def __truediv__(self, other: int|float):
        return Color(self.r / other, self.g / other, self.b / other)

    def __lshift__(self, other):
        return Color(self.r << other, self.g << other, self.b << other)

    def __rshift__(self, other):
        return Color(self.r >> other, self.g >> other, self.b >> other)

    def __eq__(self, other: 'Color'):
        return self.r == other.r and self.g == other.g and self.b == other.b

    def __ne__(self, other: 'Color'):
        return not self == other

    def __hash__(self):
        return hash((self.r, self.g, self.b))

    def __repr__(self):
        return f"#{int(self.r):0{2}X}{int(self.g):0{2}X}{int(self.b):0{2}X}"


class Point2D:
    def __init__(self, x: int | float, y: int | float, color: 'Color' = Color(255, 255, 255)):
        self.x: int = x
        self.y: int = y
        self.color: Color = color
        return

    def __eq__(self, other: 'Point2D'):
        return self.x == other.x and self.y == other.y and self.color == other.color

    def __ne__(self, other: 'Point2D'):
        return not self == other

    def __hash__(self):
        return hash((self.x, self.y, self.color))

    def __repr__(self):
        return f"({self.x}, {self.y})"

    @staticmethod
    def from_tuple(t: tuple[int | float, ...]):
        if len(t) == 2:
            return Point2D(t[0], t[1])
        elif len(t) >= 3:
            return Point3D(t[0], t[1], t[2])
        else:
            assert False

    def __add__(self, other: 'Point2D'):
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point2D'):
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other: int | float):
        return Point2D(self.x * other, self.y * other)

    def __lshift__(self, other: int):
        return Point2D(self.x << other, self.y << other)

    def __rshift__(self, other: int):
        return Point2D(self.x >> other, self.y >> other)

    def distance(self, other: 'Point2D') -> int | float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def manhattan_distance(self, other: 'Point2D') -> int | float:
        return abs(self.x - other.x) + abs(self.y - other.y)


class Line2D:
    def __init__(self, a: Point2D, b: Point2D):
        self.a: Point2D = a
        self.b: Point2D = b
        return

    def __repr__(self):
        return f"({self.a}, {self.b})"

    def length(self) -> float:
        return self.a.distance(self.b)

    def manhattan_length(self) -> float:
        return self.a.manhattan_distance(self.b)

    def pixels(self) -> set[Point2D]:
        """Returns a set of all points that make up the line"""

        def find_factor(d: Point2D) -> int:
            f = 0
            while (1 << f) < abs(d.x) or (1 << f) < abs(d.y):
                f += 1
            return f

        def correct_point(p) -> bool:
            nonlocal vertical_line, prev_p, factor
            return (vertical_line and prev_p.y != (p.y >> factor)) or (not vertical_line and prev_p.x != (p.x >> factor))

        pixels = set()

        if self.a.x > self.b.x:
            self.a, self.b = self.b, self.a

        d = self.b - self.a
        d_color = (self.b.color - self.a.color)
        vertical_line = abs(d.x) < abs(d.y)

        factor = find_factor(d)

        half = Point2D(1, 1) << (factor - 1)
        end = (self.b << factor) + half
        p = (self.a << factor) + half
        color = (self.a.color << factor) + (Color(1, 1, 1) << (factor - 1))

        while p != end:
            prev_p = p >> factor
            prev_p.color = color >> factor
            color += d_color
            p += d
            if correct_point(p):
                pixels.add(prev_p)

        pixels.add(self.b)

        return pixels



# 3D space objects

class Point3D(Point2D):
    def __init__(self, x: int | float, y: int | float, z: int | float, color: Color = Color(255, 255, 255)):
        super().__init__(x, y, color)
        self.z: int|float = z
        return

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __eq__(self, other: 'Point3D'):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other: 'Point3D'):
        return not self == other

    def __add__(self, other: 'Point3D'):
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point3D'):
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: int|float):
        return Point3D(self.x * other, self.y * other, self.z * other)

    def distance(self, other: 'Point3D') -> int | float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5

    def manhattan_distance(self, other: 'Point3D') -> int | float:
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)

    def project(self, projector: 'Projector') -> 'Point2D':
        return projector.project(self)

    def render(self, projector: 'Projector'):
        """Projects and draws the point on the screen"""
        p = self.project(projector)
        projector.screen.draw_pixel(p)
        return


class Vector(Point3D):
    def __init__(self, x, y, z = 0):
        super().__init__(x, y, z)
        return

    def dot(self, other: 'Vector') -> int|float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector') -> 'Vector':
        return Vector(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)


class Line3D:
    def __init__(self, a: Point3D, b: Point3D):
        self.a: Point3D = a
        self.b: Point3D = b
        return

    def __repr__(self):
        return f"({self.a}, {self.b})"

    def length(self) -> float:
        return self.a.distance(self.b)

    def manhattan_length(self) -> float:
        return self.a.manhattan_distance(self.b)

    def project(self, projector: 'Projector'):
        return Line2D(self.a.project(projector), self.b.project(projector))

    def render(self, projector: 'Projector'):
        """Projects and draws the point on the screen"""
        l = self.project(projector)
        projector.screen.draw_pixels(l.pixels())
        return


class Polygon:
    def __init__(self, points: list[Point3D], lines: list[tuple[int, int]]):
        self.points: list[Point3D] = points
        self.point_pairs: list[tuple[int, int]] = lines
        return

    def __repr__(self):
        return f"{self.lines()}"

    def lines(self) -> list[Line3D]:
        lines = []
        for t in self.point_pairs:
            lines.append(Line3D(self.points[t[0]], self.points[t[1]]))

        return lines

    def pixels(self, projector: 'Projector') -> set[Point2D]:
        """Returns a set of all pixels that make up the polygon's edges"""
        pixels = set()
        for line in self.project(projector):
            pixels.update(line.pixels())

        return pixels

    def project(self, projector: 'Projector') -> list[Line2D]:
        lines = []
        for line in self.lines():
            lines.append(line.project(projector))

        return lines


class Triangle(Polygon):
    pass


class Quad(Polygon):
    pass


class Solid:
    pass



# Screen

class Screen:
    def __init__(self, width: int, height: int, pixel_size: int, title: str):
        self.width = width
        self.height = height
        self.pixel_size = pixel_size
        self.title = title
        self.screen = display.set_mode((self.width * self.pixel_size, self.height * self.pixel_size))
        display.set_caption(self.title)
        return

    def update(self):
        display.flip()
        self.screen.fill((0, 0, 0))
        return

    @staticmethod
    def in_bounds(x: int, y: int, width: int, height: int):
        return 0 <= x < width and 0 <= y < height

    def draw_pixel(self, p: Point2D):
        coords = (p.x + (self.width >> 1), (self.height >> 1) - p.y)  # center the screen
        # up is positive y, right is positive x

        if self.in_bounds(coords[0], coords[1], self.width, self.height):
            pixel = pygame.Rect(coords[0] * self.pixel_size, coords[1] * self.pixel_size, self.pixel_size, self.pixel_size)
            pygame.draw.rect(self.screen, p.color.tuple(), pixel)
        return

    def draw_pixels(self, pixels: set[Point2D]):
        for p in pixels:
            self.draw_pixel(p)
        return



# Exceptions

class BehindCamera(Exception):
    pass



# Projectors

class Projector:    # default is orthographic projection
    def __init__(self, screen: Screen, pixels_per_unit: int = 500):
        self.width: float = screen.width / pixels_per_unit    # dimensions of the camera in units
        self.height: float = screen.height / pixels_per_unit  # dimensions of the camera in units
        self.pixel_density: int = pixels_per_unit
        self.screen: Screen = screen
        return

    def project(self, p: Point3D) -> Point2D:
        if self.behind_camera(p):
            raise BehindCamera
        return Point2D(p.x, p.y, p.color)

    @staticmethod
    def behind_camera(p: Point3D) -> bool:
        return p.z < 0


class PerspectiveProjector(Projector):
    def __init__(self, screen: Screen, pixel_density: int = 4444, fov: float = math.pi / 2):
        super().__init__(screen, pixel_density)
        self.fov = fov
        self.focal_length: float = self.get_focal_length(fov)
        return

    def get_focal_length(self, fov):
        return (self.width / 2) / math.tan(fov / 2)

    def project(self, p: Point3D) -> Point2D:
        if self.behind_camera(p):
            raise BehindCamera

        x = (p.x * self.focal_length) / (p.z + self.focal_length)
        y = (p.y * self.focal_length) / (p.z + self.focal_length)
        return Point2D(round(x * self.pixel_density), round(y * self.pixel_density), p.color)





# Code

def smt(screen: Screen):
    a = Point3D(0, 0, 2, Color(255, 0, 0))
    b = Point3D(1, 0, 2, Color(0, 255, 0))
    c = Point3D(1, 1, 2, Color(0, 0, 255))

    triangle = Polygon([a, b, c], [(0, 1), (0, 2), (1, 2)])

    projector = PerspectiveProjector(screen, 4444, math.pi / 2)

    screen.draw_pixels(triangle.pixels(projector))
    return


def main():
    screen_width = 100  # in pixels
    screen_height = 100  # in pixels
    pixel_size = 8

    # check to see if screen gets too big
    if screen_width * pixel_size > 1920 or screen_height * pixel_size > 1080:
        print('Screen size too big')
        exit(1)

    screen = Screen(screen_width, screen_height, pixel_size, '3D Renderer')

    clock = time.Clock()
    running = True
    while running:
        for e in event.get():
            if e.type == QUIT:
                running = False

        smt(screen)

        screen.update()

        clock.tick(10)
    return None


if __name__ == '__main__':
    main()
