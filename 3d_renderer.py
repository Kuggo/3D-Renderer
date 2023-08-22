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
        a = self.a
        b = self.b
        dx = abs(b.x - a.x)
        dy = abs(b.y - a.y)
        steep = dy > dx

        # reflect over y = x
        if steep:
            a.x, a.y = a.y, a.x
            b.x, b.y = b.y, b.x
            dx, dy = dy, dx

        # swap points   (180 rotation around the origin)
        if a.x > b.x:
            a, b = b, a

        if dx == 0:
            return {self.a}

        # line goes either up or down
        step = 1 if a.y < b.y else -1
        color_step = (b.color - a.color) / dx

        pixels = set()

        color = self.a.color
        p = (dy << 1) - dx
        y = a.y
        for x in range(a.x, b.x + 1):
            if steep:
                pixels.add(Point2D(y, x, color.round()))
            else:
                pixels.add(Point2D(x, y, color.round()))

            if p >= 0:
                y += step
                p += (dy - dx) << 1
            else:
                p += dy << 1

            color += color_step
        print(pixels)
        return pixels

    def pixels_old(self) -> set[Point2D]:
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

    def normalize(self) -> 'Vector':
        return self * (1 / self.magnitude())

    def magnitude(self) -> int|float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def angle(self, other: 'Vector') -> int|float:
        return math.acos(self.dot(other) / (self.magnitude() * other.magnitude()))

    def rotate_pitch(self, pitch: float):
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        x = self.x * cos_pitch - self.z * sin_pitch
        z = self.x * sin_pitch + self.z * cos_pitch
        return Vector(x, self.y, z)

    def rotate_yaw(self, yaw: float):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x = self.x * cos_yaw - self.z * sin_yaw
        z = self.x * sin_yaw + self.z * cos_yaw
        return Vector(x, self.y, z)

    def rotate_roll(self, roll: float):
        cos_roll = math.cos(roll)
        sin_roll = math.sin(roll)
        x = self.x * cos_roll - self.y * sin_roll
        y = self.x * sin_roll + self.y * cos_roll
        return Vector(x, y, self.z)


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
        print(lines)
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

        left_bound = -(self.width >> 1)
        right_bound = (self.width >> 1) - 1
        bottom_bound = -(self.height >> 1)
        top_bound = (self.height >> 1) - 1

        intersecting_points = []  # finding at most 2 points of intersection with the edge of the screen
        if line.a.x != line.b.x:  # not vertical line
            if line.a.x < left_bound <= line.b.x or line.b.x < left_bound <= line.a.x:  # left y-axis
                k = inv_lerp(line.a.x, line.b.x, left_bound)
                y = lerp(line.a.y, line.b.y, k)
                if bottom_bound <= y <= top_bound:
                    color = lerp(line.a.color, line.b.color, k)
                    intersecting_points.append(Point2D(left_bound, y, color.round()))

            if line.a.x <= right_bound < line.b.x or line.b.x <= right_bound < line.a.x:  # right y-axis
                k = inv_lerp(line.a.x, line.b.x, right_bound)
                y = lerp(line.a.y, line.b.y, k)
                if bottom_bound <= y <= top_bound:
                    color = lerp(line.a.color, line.b.color, k)
                    intersecting_points.append(Point2D(right_bound, y, color.round()))

        if line.a.y != line.b.y:  # not horizontal line
            if line.a.y < bottom_bound <= line.b.y or line.b.y < bottom_bound <= line.a.y:  # bottom x-axis
                k = inv_lerp(line.a.y, line.b.y, bottom_bound)
                x = lerp(line.a.x, line.b.x, k)
                if left_bound <= x <= right_bound:
                    color = lerp(line.a.color, line.b.color, k)
                    intersecting_points.append(Point2D(x, bottom_bound, color.round()))

            if line.a.y <= top_bound < line.b.y or line.b.y <= top_bound < line.a.y:  # top x-axis
                k = inv_lerp(line.a.y, line.b.y, top_bound)
                x = lerp(line.a.x, line.b.x, k)
                if left_bound <= x <= right_bound:
                    color = lerp(line.a.color, line.b.color, k)
                    intersecting_points.append(Point2D(x, top_bound, color.round()))

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
        return -(self.width >> 1) <= point.x < (self.width >> 1) and -(self.height >> 1) <= point.y < (self.height >> 1)

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

        # lut values
        self.sin_pitch = math.sin(self.pitch)
        self.cos_pitch = math.cos(self.pitch)
        self.sin_yaw = math.sin(self.yaw)
        self.cos_yaw = math.cos(self.yaw)
        self.sin_roll = math.sin(self.roll)
        self.cos_roll = math.cos(self.roll)
        return

    def update_lut(self):
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
        self.update_lut()
        return

    def rotate_point(self, p: Point3D) -> Point3D:
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

    def get_direction(self) -> Vector:
        """Returns a unit vector pointing in the direction the camera is facing"""
        #return Vector(math.cos(self.yaw) * math.cos(self.pitch),
        #             math.sin(self.yaw) * math.cos(self.pitch),
        #             math.sin(self.pitch))

        return Vector(math.cos(self.yaw) * math.cos(self.pitch),
                      -math.sin(self.yaw) * math.cos(self.pitch),
                      math.sin(self.pitch))

    def set_direction(self, direction_vector: Vector):
        self.yaw = math.atan2(direction_vector.y, direction_vector.x)

        self.pitch = math.atan2(-direction_vector.z, math.sqrt(direction_vector.x ** 2 + direction_vector.y ** 2))

        self.roll = math.atan2(direction_vector.x, direction_vector.z)
        return

    def world_to_camera(self, p: Point3D) -> Point3D:
        return self.rotate_point(p - self.position)

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

    camera.rotate_pov(pi_div_by_50/2, pi_div_by_50, 0)
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
