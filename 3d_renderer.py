import math
from typing import Optional

import pygame.draw
from pygame import display, event, QUIT, time

# auxiliary functions

def lerp(a, b, t):
    return a + (b - a) * t


def inv_lerp(a, b, v):
    return (v - a) / (b - a)


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

        pixels = set()

        if self.a.x > self.b.x:
            self.a, self.b = self.b, self.a

        d = self.b - self.a
        d_color = (self.b.color - self.a.color)
        vertical_line = abs(d.x) < abs(d.y)

        factor = find_factor(d)
        if factor == 0:
            pixels.add(self.a)
            pixels.add(self.b)
            return pixels

        assert factor > 0

        half = Point2D(1, 1) << (factor - 1)
        end = (self.b << factor) + half
        p = (self.a << factor) + half
        color = (self.a.color << factor) + (Color(1, 1, 1) << (factor - 1))

        while p != end:
            prev_p = p >> factor
            prev_p.color = color >> factor
            color += d_color
            p += d
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
        # checking if line intersects camera plane
        point = projector.intersect_line_camera(self)
        if point is not None:
            if projector.behind_camera(self.a):
                return Line2D(point, self.b.project(projector))
            else:
                return Line2D(self.a.project(projector), point)

        return Line2D(self.a.project(projector), self.b.project(projector))

    def render(self, projector: 'Projector'):
        """Projects and draws the point on the screen"""
        l2d = self.project(projector)
        l_culled = projector.screen.line_culling(l2d)
        projector.screen.draw_pixels(l_culled.pixels())
        return


class Polygon:
    def __init__(self, points: list[Point3D], lines: list[tuple[int, int]]):
        self.points: list[Point3D] = points
        self.lines = self.lines(points, lines)
        self.point_pairs: list[tuple[int, int]] = lines
        return

    def __repr__(self):
        return f"{self.lines}"

    @staticmethod
    def lines(points, point_pairs) -> list[Line3D]:
        lines = []
        for t in point_pairs:
            lines.append(Line3D(points[t[0]], points[t[1]]))

        return lines

    def pixels(self, projector: 'Projector') -> set[Point2D]:
        """Returns a set of all pixels that make up the polygon's edges"""
        pixels = set()
        for line in self.project(projector):
            pixels.update(line.pixels())

        return pixels

    def project(self, projector: 'Projector') -> list[Line2D]:
        lines = []
        for line in self.lines:
            lines.append(line.project(projector))

        return lines

    def render(self, projector: 'Projector'):
        """Projects and draws the polygon on the screen"""
        pixels = self.pixels(projector)
        projector.screen.draw_pixels(pixels)
        return


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

    def line_culling(self, line: Line2D) -> Optional[Line2D]:
        """Returns the portion of the line that is visible, None if it's not visible at all"""
        if self.in_bounds(line.a) and self.in_bounds(line.b):
            return line

        w2 = self.width >> 1
        h2 = self.height >> 1

        intersecting_points = []  # finding at most 2 points of intersection with the edge of the screen
        if line.a.x != line.b.x:  # not vertical line
            if line.a.x < -w2 < line.b.x or line.b.x < -w2 < line.a.x:  # left y-axis
                k = inv_lerp(line.a.x, line.b.x, -w2)
                y = lerp(line.a.y, line.b.y, k)
                if -h2 <= y <= h2:
                    color = lerp(line.a.color, line.b.color, k)
                    intersecting_points.append(Point2D(-w2, y, color.round()))

            if line.a.x < w2 < line.b.x or line.b.x < w2 < line.a.x:  # right y-axis
                k = inv_lerp(line.a.x, line.b.x, w2)
                y = lerp(line.a.y, line.b.y, k)
                if -h2 <= y <= h2:
                    color = lerp(line.a.color, line.b.color, k)
                    intersecting_points.append(Point2D(w2, y, color.round()))

        if line.a.y != line.b.y:  # not horizontal line
            if line.a.y < -h2 < line.b.y or line.b.y < -h2 < line.a.y:  # bottom x-axis
                k = inv_lerp(line.a.y, line.b.y, -h2)
                x = lerp(line.a.x, line.b.x, k)
                if -w2 <= x <= w2:
                    color = lerp(line.a.color, line.b.color, k)
                    intersecting_points.append(Point2D(x, -h2, color.round()))

            if line.a.y < h2 < line.b.y or line.b.y < h2 < line.a.y:  # top x-axis
                k = inv_lerp(line.a.y, line.b.y, h2)
                x = lerp(line.a.x, line.b.x, k)
                if -w2 <= x <= w2:
                    color = lerp(line.a.color, line.b.color, k)
                    intersecting_points.append(Point2D(x, h2, color.round()))

        assert 0 < len(intersecting_points) <= 2

        if len(intersecting_points) == 2:
            return Line2D(intersecting_points[0], intersecting_points[1])

        if self.in_bounds(line.a):  # b must be out of bounds
            return Line2D(line.a, intersecting_points[0])
        else:  # a must be out of bounds
            return Line2D(intersecting_points[0], line.b)

    def in_bounds(self, point: Point2D):
        return -(self.width >> 1) <= point.x < (self.width >> 1) and -(self.height >> 1) < point.y <= (self.height >> 1)

    def center_pixel(self, point: Point2D):
        # up is positive y, right is positive x
        return Point2D(point.x + (self.width >> 1), (self.height >> 1) - point.y, point.color)

    def draw_pixel(self, p: Point2D):
        if self.in_bounds(p):
            coords = self.center_pixel(p) * self.pixel_size
            pixel = pygame.Rect(coords.x, coords.y, self.pixel_size, self.pixel_size)
            pygame.draw.rect(self.screen, p.color.tuple(), pixel)
        else:
            raise NotVisibleError
        return

    def draw_pixels(self, pixels: set[Point2D]):
        for p in pixels:
            try:
                self.draw_pixel(p)
            except NotVisibleError:
                pass
        return



# Exceptions

class BehindCameraError(Exception):
    pass


class NotVisibleError(Exception):
    pass



# Projectors

class Projector:    # default is orthographic projection
    def __init__(self, screen: Screen, pixels_per_unit: int = 555, camera: Point3D = Point3D(0, 0, 0)):
        self.width: float = screen.width / pixels_per_unit    # dimensions of the camera in units
        self.height: float = screen.height / pixels_per_unit  # dimensions of the camera in units
        self.pixels_per_unit: int = pixels_per_unit
        self.screen: Screen = screen
        self.camera: Point3D = camera
        self.pitch: float = 0   # positive is up
        self.yaw: float = 0     # positive is right
        self.roll: float = 0    # positive is ??
        return

    def project(self, p: Point3D) -> Point2D:
        if self.behind_camera(p):
            raise BehindCameraError

        p_moved = p - self.camera

        p_rotated = self.rotate(p_moved)
        return Point2D(round(p_rotated.x * self.pixels_per_unit), round(p_rotated.y * self.pixels_per_unit), p.color)

    def rotate(self, p: Point3D) -> Point3D:
        x = self.rotate_for_x(p)
        y = self.rotate_for_y(p)
        z = self.rotate_for_z(p)
        return Point3D(x, y, z, p.color)

    def rotate_for_x(self, p: Point3D):  # rotation matrix's yaw = y axis, pitch = x axis, roll = y axis
        x = (p.x * math.cos(self.roll)) - (p.y * math.sin(self.roll))
        return (x * math.cos(self.yaw)) + (p.z * math.sin(self.yaw))

    def rotate_for_y(self, p: Point3D):
        y = (p.x * math.sin(self.roll)) + (p.y * math.cos(self.roll))
        return (y * math.cos(self.pitch)) + (p.z * math.sin(self.pitch))

    def rotate_for_z(self, p: Point3D):
        z = -(p.y * math.sin(self.pitch)) + (p.z * math.cos(self.pitch))
        return (z * math.cos(self.yaw)) - (p.x * math.sin(self.yaw))

    def intersect_line_camera(self, line: Line3D) -> Optional[Point3D]:
        """Returns the point where the line intersects the camera plane, or None if it doesn't intersect,
        or it's contained on the place"""

        if self.behind_camera(line.a) ^ self.behind_camera(line.b):
            k = -line.a.z / (line.b.z - line.a.z)   # no div by zero for sure
            color = line.a.color + (line.b.color - line.a.color) * k
            return Point3D((k-1) * (line.b.x - line.a.x), (k-1) * (line.b.y - line.a.y), 0, color)

        return None

    def behind_camera(self, p: Point3D) -> bool:
        return p.z < 0


class PerspectiveProjector(Projector):
    def __init__(self, screen: Screen, pixel_density: int = 555, fov: float = math.pi / 2, camera: Point3D = Point3D(0, 0, 0)):
        super().__init__(screen, pixel_density, camera)
        self.fov = fov
        self.focal_length: float = self.get_focal_length(fov)
        return

    def get_focal_length(self, fov):
        return (self.width / 2) / math.tan(fov / 2)

    def intersect_line_camera(self, line: Line3D) -> Optional[Point3D]:
        """Returns the point where the line intersects the camera plane, or None if it doesn't intersect,
        or it's contained on the place"""

        if self.behind_camera(line.a) ^ self.behind_camera(line.b):
            k = (self.focal_length - line.a.z) / (line.b.z - line.a.z)  # no div by zero for sure
            color = line.a.color * (1 - k) + line.b.color * k
            return Point3D((k - 1) * (line.b.x - line.a.x), (k - 1) * (line.b.y - line.a.y), self.focal_length, color)

        return None

    def behind_camera(self, p: Point3D) -> bool:
        return p.z < self.focal_length

    def project(self, p: Point3D) -> Point2D:
        if self.behind_camera(p):
            raise BehindCameraError

        p_moved = p - self.camera

        p_rotated = self.rotate(p_moved)

        x = (p_rotated.x * self.focal_length) / p_rotated.z
        y = (p_rotated.y * self.focal_length) / p_rotated.z
        return Point2D(round(x * self.pixels_per_unit), round(y * self.pixels_per_unit), p.color)





# Code

def smt(projector: Projector, i):
    pi_div_by_50 = math.pi * 2 / 100
    # fl = projector.focal_length

    a = Point3D(-1, -1, 2)
    b = Point3D(-1, -1, 4)
    c = Point3D(1, -1, 2)
    d = Point3D(1, -1, 4)
    e = Point3D(-1, 1, 2)
    f = Point3D(-1, 1, 4)
    g = Point3D(1, 1, 2)
    h = Point3D(1, 1, 4)

    connections = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

    cube = Polygon([a, b, c, d, e, f, g, h], connections)

    cube.render(projector)
    projector.yaw = i * pi_div_by_50

    # screen.draw_pixels(triangle.pixels(projector))

    l = Line2D(Point2D(0, 10), Point2D(11, 0)).pixels()
    #projector.screen.draw_pixels(l)
    return


def main():
    screen_width = 100  # in pixels
    screen_height = 100  # in pixels
    pixel_size = 8
    fps = 5

    # check to see if screen gets too big
    if screen_width * pixel_size > 1920 or screen_height * pixel_size > 1080:
        print('Screen size too big')
        exit(1)

    screen = Screen(screen_width, screen_height, pixel_size, '3D Renderer')
    projector = PerspectiveProjector(screen, 555, math.pi / 2)
    iteration_n = 0
    clock = time.Clock()
    running = True
    while running:
        for e in event.get():
            if e.type == QUIT:
                running = False

        smt(projector, iteration_n)

        screen.update()
        iteration_n += 1
        clock.tick(fps)
    return None


if __name__ == '__main__':
    main()
