import math
from typing import Optional

import pygame.draw
from pygame import display, event, QUIT, time

# auxiliary functions

def lerp(a, b, t):
    return a + (b - a) * t


def inv_lerp(a, b, v):
    return (v - a) / (b - a)


def clamp(a, b, v):
    return min(b, max(a, v))



# Exceptions

class BehindCameraError(Exception):
    pass


class NotVisibleError(Exception):
    pass



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
        return clamp(0, 255, round(self.r)), clamp(0, 255, round(self.g)), clamp(0, 255, round(self.b))

    def round(self):
        return Color(clamp(0, 255, round(self.r)),
                     clamp(0, 255, round(self.g)),
                     clamp(0, 255, round(self.b)))

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

    def __round__(self, n=None):
        return Color(round(self.r, n), round(self.g, n), round(self.b, n))

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
        return self.x == other.x and self.y == other.y

    def __ne__(self, other: 'Point2D'):
        return not self == other

    def __hash__(self):
        return hash((self.x, self.y, self.color))

    def __repr__(self):
        return f"({self.x}, {self.y})"

    @staticmethod
    def from_tuple(t: tuple[int | float, ...]):
        if len(t) == 2:
            return Point2D(t[0], t[1], color=Color(255, 255, 255))
        elif len(t) >= 3:
            return Point3D(t[0], t[1], t[2], color=Color(255, 255, 255))
        else:
            assert False

    def __add__(self, other: 'Point2D'):
        return Point2D(self.x + other.x, self.y + other.y, self.color)

    def __sub__(self, other: 'Point2D'):
        return Point2D(self.x - other.x, self.y - other.y, self.color)

    def __mul__(self, other: int | float):
        return Point2D(self.x * other, self.y * other, self.color)

    def __lshift__(self, other: int):
        return Point2D(self.x << other, self.y << other, self.color)

    def __rshift__(self, other: int):
        return Point2D(self.x >> other, self.y >> other, self.color)

    def __round__(self, n=None):
        return Point2D(round(self.x, n), round(self.y, n), self.color)

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
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z, self.color)

    def __sub__(self, other: 'Point3D'):
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z, self.color)

    def __mul__(self, other: int|float):
        return Point3D(self.x * other, self.y * other, self.z * other, self.color)

    def __round__(self, n=None):
        return Point3D(round(self.x, n), round(self.y, n), round(self.z, n))

    def distance(self, other: 'Point3D') -> int | float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5

    def manhattan_distance(self, other: 'Point3D') -> int | float:
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)

    def project(self, camera: 'Camera') -> 'Point2D':
        camera_p = camera.world_to_camera(self)
        return camera.projector.project(camera_p)

    def render(self, camera: 'Camera'):
        """Projects and draws the point on the screen"""
        p = self.project(camera)
        camera.projector.screen.draw_pixel(p)
        return


class Vector(Point3D):
    def __init__(self, x, y, z = 0):
        super().__init__(x, y, z)
        return

    def dot(self, other: 'Vector') -> int|float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector') -> 'Vector':
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)


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

    def project(self, camera: 'Camera') -> 'Line2D':
        a = camera.world_to_camera(self.a)
        b = camera.world_to_camera(self.b)
        l = Line3D(a, b)

        # checking if line intersects camera plane
        point = camera.projector.intersect_line_camera(l)
        if point is not None:
            if camera.projector.behind_plane(a):
                return Line2D(camera.projector.project(point), camera.projector.project(b))
            else:
                return Line2D(camera.projector.project(a), camera.projector.project(point))

        elif camera.projector.behind_plane(a) and camera.projector.behind_plane(b):
            raise BehindCameraError

        else:
            return Line2D(camera.projector.project(a), camera.projector.project(b))

    def render(self, camera: 'Camera'):
        """Projects and draws the point on the screen"""
        try:
            l2d = self.project(camera)
        except BehindCameraError:
            return
        l_culled = camera.projector.screen.line_culling(l2d)
        if l_culled is None:
            return

        camera.projector.screen.draw_pixels(l_culled.pixels())
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

    def pixels(self, camera: 'Camera') -> set[Point2D]:
        """Returns a set of all pixels that make up the polygon's edges"""
        pixels = set()
        for line in self.project(camera):
            l_culled = camera.projector.screen.line_culling(line)
            if l_culled is None:
                continue
            pixels.update(l_culled.pixels())

        return pixels

    def project(self, camera: 'Camera') -> list[Line2D]:
        lines = []
        for line in self.lines:
            try:
                l2d = line.project(camera)
            except BehindCameraError:
                continue
            lines.append(l2d)

        return lines

    def render(self, camera: 'Camera'):
        """Projects and draws the polygon on the screen"""
        pixels = self.pixels(camera)
        camera.projector.screen.draw_pixels(pixels)
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

        assert 0 <= len(intersecting_points) <= 2

        if not self.in_bounds(line.a) and not self.in_bounds(line.b):
            return None

        if len(intersecting_points) == 2:
            return Line2D(intersecting_points[0].__round__(), intersecting_points[1].__round__())

        if self.in_bounds(line.a):  # b must be out of bounds
            return Line2D(line.a, intersecting_points[0].__round__())
        else:  # a must be out of bounds
            return Line2D(intersecting_points[0].__round__(), line.b)

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



# Camera Orientation

class Camera:
    def __init__(self, projector: 'Projector', position: Point3D = Point3D(0, 0, 0),
                 pitch: float = 0, yaw: float = 0, roll: float = 0):
        self.projector = projector
        self.position: Point3D = position
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

        self.direction: Vector = Vector(0, 0, 1)

        # lut values
        self.sin_pitch = math.sin(self.pitch)
        self.cos_pitch = math.cos(self.pitch)
        self.sin_yaw = math.sin(self.yaw)
        self.cos_yaw = math.cos(self.yaw)
        self.sin_roll = math.sin(self.roll)
        self.cos_roll = math.cos(self.roll)

        self.change_direction()
        return

    def change_direction(self):
        #self.direction = self.rotate_pitch(self.direction)
        #self.direction = self.rotate_yaw(self.direction)
        #self.direction = self.rotate_roll(self.direction)

        self.sin_pitch = math.sin(self.pitch)
        self.cos_pitch = math.cos(self.pitch)
        self.sin_yaw = math.sin(self.yaw)
        self.cos_yaw = math.cos(self.yaw)
        self.sin_roll = math.sin(self.roll)
        self.cos_roll = math.cos(self.roll)
        return

    def rotate_pov(self, pitch: float = 0, yaw: float = 0, roll: float = 0):
        self.pitch += pitch
        self.yaw += yaw
        self.roll += roll
        self.change_direction()
        return

    """def rotate_point(self, p: Point3D) -> Point3D:
        v = Vector(p.x, p.y, p.z)

        x = v.dot(self.direction)
        y = v.dot(Vector(-self.direction.z, self.direction.z, self.direction.x))
        z = v.dot(Vector(-self.direction.y, self.direction.x, self.direction.z))
        return Point3D(x, y, z, p.color)
    """
    def rotate_point1(self, p: Point3D) -> Point3D:
        # rotation matrix's yaw = y axis, pitch = x axis, roll = y axis
        p = self.rotate_roll(p)
        p = self.rotate_pitch(p)
        p = self.rotate_yaw(p)
        return p

    def rotate_roll(self, p: Point3D):
        x = p.x * self.cos_roll - p.y * self.sin_roll
        y = p.x * self.sin_roll + p.y * self.cos_roll
        return Point3D(x, y, p.z, p.color)

    def rotate_pitch(self, p: Point3D):
        y = p.y * self.cos_pitch - p.z * self.sin_pitch
        z = p.y * self.sin_pitch + p.z * self.cos_pitch
        return Point3D(p.x, y, z, p.color)

    def rotate_yaw(self, p: Point3D):
        # when yaw = 0, x-axis is forward and z-axis is to the left
        # z = p.x * self.cos_yaw + p.z * self.sin_yaw
        # x = -p.x * self.sin_yaw + p.z * self.cos_yaw

        # when yaw = 0, z-axis is forward and x-axis is to the right
        z = -p.x * self.sin_yaw + p.z * self.cos_yaw
        x = -p.x * self.cos_yaw - p.z * self.sin_yaw
        return Point3D(x, p.y, z, p.color)

    def set_direction(self, direction: Vector):
        self.direction = direction

        self.yaw = math.atan2(self.direction.y, self.direction.x)

        self.pitch = math.atan2(-self.direction.z, math.sqrt(self.direction.x ** 2 + self.direction.y ** 2))

        self.roll = math.atan2(self.direction.x, self.direction.z)

        self.sin_pitch = math.sin(self.pitch)
        self.cos_pitch = math.cos(self.pitch)
        self.sin_yaw = math.sin(self.yaw)
        self.cos_yaw = math.cos(self.yaw)
        self.sin_roll = math.sin(self.roll)
        self.cos_roll = math.cos(self.roll)
        return

    def world_to_camera(self, p: Point3D) -> Point3D:
        return self.rotate_point1(p - self.position)

    def behind_camera(self, p: Point3D) -> bool:
        return self.projector.behind_plane(self.world_to_camera(p))



# Projectors

class Projector:    # default is orthographic projection
    def __init__(self, screen: Screen, pixels_per_unit: int = 555):
        self.width: float = screen.width / pixels_per_unit    # dimensions of the camera in units
        self.height: float = screen.height / pixels_per_unit  # dimensions of the camera in units
        self.pixels_per_unit: int = pixels_per_unit
        self.screen: Screen = screen
        return

    def project(self, p: Point3D) -> Point2D:
        if self.behind_plane(p):
            raise BehindCameraError

        return self.world_to_pixels(p)

    def world_to_pixels(self, p: Point2D) -> Point2D:
        return Point2D(round(p.x * self.pixels_per_unit), round(p.y * self.pixels_per_unit), p.color)


    def intersect_line_camera(self, line: Line3D) -> Optional[Point3D]:
        """Returns the point where the line intersects the camera plane, or None if it doesn't intersect,
        or it's contained on the place"""

        if self.behind_plane(line.a) ^ self.behind_plane(line.b):
            k = inv_lerp(line.a.z, line.b.z, 0)     # no div by zero for sure
            color = lerp(line.a.color, line.b.color, k).__round__()
            return Point3D(lerp(line.a.x, line.b.x, k), lerp(line.a.y, line.b.y, k), 0, color)

        return None

    def behind_plane(self, p: Point3D) -> bool:
        return p.z < 0


class PerspectiveProjector(Projector):
    def __init__(self, screen: Screen, pixel_density: int = 555, fov: float = math.pi / 2):
        super().__init__(screen, pixel_density)
        self.fov = fov
        self.focal_length: float = self.get_focal_length(fov)
        return

    def get_focal_length(self, fov):
        return (self.width / 2) / math.tan(fov / 2)

    def intersect_line_camera(self, line: Line3D) -> Optional[Point3D]:
        """Returns the point where the line intersects the camera plane, or None if it doesn't intersect,
        or it's contained on the place"""

        if self.behind_plane(line.a) ^ self.behind_plane(line.b):
            k = inv_lerp(line.a.z, line.b.z, self.focal_length)  # no div by zero for sure
            color = lerp(line.a.color, line.b.color, k).__round__()
            return Point3D(lerp(line.a.x, line.b.x, k), lerp(line.a.y, line.b.y, k), self.focal_length, color)

        return None

    def behind_plane(self, p: Point3D) -> bool:
        return p.z < self.focal_length

    def project(self, p: Point3D) -> Point2D:
        if self.behind_plane(p):
            raise BehindCameraError

        x = (p.x * self.focal_length) / p.z
        y = (p.y * self.focal_length) / p.z
        return Point2D(round(x * self.pixels_per_unit), round(y * self.pixels_per_unit), p.color)





# Code

def smt(camera: Camera, i):
    pi_div_by_50 = 2 * math.pi / 100

    a = Point3D(-1, -1, 2, Color(255, 0, 0))
    b = Point3D(-1, -1, 4, Color(0, 255, 0))
    c = Point3D(1, -1, 2, Color(0, 0, 255))
    d = Point3D(1, -1, 4, Color(0, 255, 0))
    e = Point3D(-1, 1, 2, Color(255, 0, 0))
    f = Point3D(-1, 1, 4, Color(0, 0, 255))
    g = Point3D(1, 1, 2, Color(0, 255, 0))
    h = Point3D(1, 1, 4, Color(255, 0, 0))

    connections = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

    cube = Polygon([a, b, c, d, e, f, g, h], connections)

    camera.rotate_pov(0, pi_div_by_50, 0)
    cube.render(camera)
    return


def debug(camera: Camera, i):
    pi_div_by_5 = math.pi * 2 / 50
    a = Point3D(-1, -1, 2, Color(255, 0, 0))
    b = Point3D(-1, -1, 4, Color(0, 255, 0))
    c = Point3D(1, -1, 2, Color(0, 0, 255))
    d = Point3D(1, -1, 4, Color(0, 255, 0))

    connections = [(0, 1), (0, 2), (1, 3), (2, 3)]

    quad = Polygon([a, b, c, d], connections)
    triangle = Polygon([a, b, c], [(0, 1), (0, 2), (1, 2)])

    camera.rotate_pov(0, pi_div_by_5, 0)
    # camera.yaw = i * pi_div_by_5
    # quad.render(projector)
    triangle.render(camera)

    return


def main():
    screen_width = 100  # in pixels
    screen_height = 100  # in pixels
    pixel_size = 8
    fps = 10

    # check to see if screen gets too big
    if screen_width * pixel_size > 1920 or screen_height * pixel_size > 1080:
        print('Screen size too big')
        exit(1)

    screen = Screen(screen_width, screen_height, pixel_size, '3D Renderer')
    projector = PerspectiveProjector(screen, 555, math.pi / 2)
    camera = Camera(projector, Point3D(0, 0, 0), 0, 0, 0)

    iteration_n = 0
    clock = time.Clock()
    running = True
    while running:
        for e in event.get():
            if e.type == QUIT:
                running = False

        smt(camera, iteration_n)
        # debug(camera, iteration_n)

        print(iteration_n)

        screen.update()
        iteration_n += 1/fps
        clock.tick(fps)
    return None


if __name__ == '__main__':
    main()
