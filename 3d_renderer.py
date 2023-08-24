import math
from typing import Optional
from os import path as os_path
import pygame.draw
import pygame as pg


# constants

fp_tolerance = 1e-6


# auxiliary functions


def fp_equals(a, b) -> bool:
    return abs(a - b) < fp_tolerance

def lerp(a, b, t):
    return a + (b - a) * t


def inv_lerp(a, b, v):
    return (v - a) / (b - a)


def clamp(a, b, v):
    return min(b, max(a, v))


def load_object_file(file_name) -> 'Mesh':
    file_path = os_path.join("objects", file_name)
    points: list[Point3D] = []
    faces = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            tokens = line.strip().split()

            if len(tokens) == 0:
                continue

            if tokens[0] == 'v':
                assert len(tokens) >= 3
                x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])

                if len(tokens) == 4:
                    points.append(Point3D(x, y, z))
                    continue

                w = float(tokens[4])

                if len(tokens) == 5:
                    continue

                r, g, b = float(tokens[5]), float(tokens[6]), float(tokens[7])
                color = (Color(r, g, b) * 255).round()
                points.append(Point3D(x, y, z, color))

            elif tokens[0] == 'vt':     # TODO implement the rest of the file specifications
                pass

            elif tokens[0] == 'vn':
                pass

            elif tokens[0] == 'f':
                face_points = []
                for vertex in tokens[1:]:
                    print(tokens)
                    indices = []
                    for i in vertex.split('/'):
                        indices.append(int(i))

                    if len(indices) == 0:
                        continue
                    vertex_index = indices[0] - 1 if indices[0] > 0 else indices[0]

                    if len(indices) == 1:
                        face_points.append(points[vertex_index])
                        continue
                    texture_index = indices[1] - 1 if indices[1] > 0 else indices[1]

                    if len(indices) == 2:
                        face_points.append(points[vertex_index])
                        continue
                    normal_index = indices[2] - 1 if indices[2] > 0 else indices[2]
                    face_points.append(points[vertex_index])
                    continue

                if len(face_points) < 3:
                    raise InvalidPolygonError("A polygon must have at least 3 vertices")
                if len(face_points) == 3:
                    face = Triangle(face_points)
                elif len(face_points) == 4:
                    face = Quad(face_points)
                else:
                    face = Polygon(face_points)
                faces.append(face)

    mesh = Mesh(faces)
    return mesh


def change_plane(points: list['Point3D'], normal: 'Vector', offset: 'Point3D') -> list['Point3D']:
    new_points = []
    for point in points:
        new_points.append(normal.use_as_reference(point) + offset)

    return new_points


def circle(radius, color, resolution: int = 8, cw=True) -> list['Point3D']:
    """Returns a list of points that make up a circle in the x y plane, centered at center"""
    if fp_equals(radius, 0):
        return [Point3D(0, 0, 0, color)]

    side_num = 4 * resolution
    angle_step = 2 * math.pi / side_num
    direction = -1 if cw else 1
    points = []

    angle = 0
    i = 0
    while i < side_num:
        x = direction * math.sin(angle) * radius
        y = math.cos(angle) * radius
        points.append(Point3D(x, y, 0, color))
        angle += angle_step
        i += 1
        continue

    return points


def create_polygons(prev_points, points, opposite_ways=False) -> list['Polygon']:
    def index(l, j):
        nonlocal div
        return l[round(j * len(l) / div) % len(l)]

    polygons = []
    div = max(len(points), len(prev_points))
    for i in range(0, max(len(points), len(prev_points)), 1):
        if opposite_ways:
            a = index(points, len(points)//2 - i)
            b = index(points, len(points)//2 - i - 1)
        else:
            a = index(points, i)
            b = index(points, i + 1)

        c = index(prev_points, i + 1)
        d = index(prev_points, i)

        if c == d:
            polygons.append(Triangle([a, b, c]))
        elif a == b:
            polygons.append(Triangle([c, d, a]))
        else:
            polygons.append(Quad([a, b, c, d]))

    return polygons



# Exceptions

class ScreenSizeError(Exception):
    pass


class BehindCameraError(Exception):
    pass


class NotVisibleError(Exception):
    pass


class InvalidPolygonError(Exception):
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

    @staticmethod
    def from_hex(hex_str: str):
        return Color(int(hex_str[1:3], 16), int(hex_str[3:5], 16), int(hex_str[5:7], 16))

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

    def __lt__(self, other: 'Point2D'):
        return self.y < other.y or (self.y == other.y and self.x < other.x)

    def __le__(self, other: 'Point3D'):
        return self < other or self == other

    def __gt__(self, other: 'Point3D'):
        return not self <= other

    def __ge__(self, other: 'Point3D'):
        return not self < other

    def __hash__(self):
        return hash((self.x, self.y))

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

    def __neg__(self):
        return Point2D(-self.x, -self.y, self.color)

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
        if a < b:
            self.a: Point2D = a
            self.b: Point2D = b
        else:
            self.a: Point2D = b
            self.b: Point2D = a
        return

    def __repr__(self):
        return f"({self.a}, {self.b})"

    def __eq__(self, other: 'Line2D'):
        return (self.a == other.a and self.b == other.b) or (self.a == other.b and self.b == other.a)

    def __hash__(self):
        return hash((self.a, self.b))

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
        return f"({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __eq__(self, other: 'Point3D'):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __ne__(self, other: 'Point3D'):
        return not self == other

    def __lt__(self, other: 'Point3D'):
        return self.x < other.x or (self.x == other.x and self.y < other.y) or \
            (self.x == other.x and self.y == other.y and self.z < other.z)

    def __le__(self, other: 'Point3D'):
        return self < other or self == other

    def __gt__(self, other: 'Point3D'):
        return not self <= other

    def __ge__(self, other: 'Point3D'):
        return not self < other

    def __add__(self, other: 'Point3D'):
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z, self.color)

    def __sub__(self, other: 'Point3D'):
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z, self.color)

    def __neg__(self):
        return Point3D(-self.x, -self.y, -self.z, self.color)

    def __mul__(self, other: int|float):
        return Point3D(self.x * other, self.y * other, self.z * other, self.color)

    def __round__(self, n=None):
        return Point3D(round(self.x, n), round(self.y, n), round(self.z, n))

    def fp_equals(self, other: 'Point3D'):
        return abs(self.x - other.x) < fp_tolerance and abs(self.y - other.y) < fp_tolerance and \
            abs(self.z - other.z) < fp_tolerance

    def distance(self, other: 'Point3D') -> int | float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5

    def manhattan_distance(self, other: 'Point3D') -> int | float:
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)

    def rotate_pitch(self, pitch: float) -> 'Point3D':
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        y = self.y * cos_pitch - self.z * sin_pitch
        z = self.y * sin_pitch + self.z * cos_pitch
        return Point3D(self.x, y, z, self.color)

    def rotate_yaw(self, yaw: float) -> 'Point3D':
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x = self.x * cos_yaw - self.z * sin_yaw
        z = self.x * sin_yaw + self.z * cos_yaw
        return Point3D(x, self.y, z, self.color)

    def rotate_roll(self, roll: float) -> 'Point3D':
        cos_roll = math.cos(roll)
        sin_roll = math.sin(roll)
        x = self.x * cos_roll - self.y * sin_roll
        y = self.x * sin_roll + self.y * cos_roll
        return Point3D(x, y, self.z, self.color)

    def project(self, camera: 'Camera') -> 'Point2D':
        camera_p = camera.world_to_camera(self)
        return camera.projector.project(camera_p)

    def render(self, camera: 'Camera'):
        """Projects and draws the point on the screen"""
        p = self.project(camera)
        camera.projector.screen.draw_pixel(p)
        return


class Vector(Point3D):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        return

    @staticmethod
    def from_points(a: Point3D, b: Point3D) -> 'Vector':
        return Vector(b.x - a.x, b.y - a.y, b.z - a.z)

    def get_polar(self) -> tuple[int|float, int|float]:
        """Returns the polar coordinates of the vector, pitch and yaw"""
        pitch = math.asin(self.y / self.magnitude())
        yaw = math.atan2(self.x, self.z)
        return pitch, yaw

    def use_as_reference(self, point: Point3D) -> Point3D:
        """Returns the vector projected on the other vector"""
        pitch, yaw = self.get_polar()
        return point.rotate_pitch(-pitch).rotate_yaw(-yaw)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: int|float):
        return Vector(self.x * other, self.y * other, self.z * other)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def dot(self, other: 'Point3D') -> int|float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector') -> 'Vector':
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)

    def colinear(self, other: 'Vector') -> bool:
        return fp_equals(self.dot(other), 0)

    def normalize(self) -> 'Vector':
        return self * (1 / self.magnitude())

    def magnitude(self) -> int|float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def angle(self, other: 'Vector') -> int|float:
        return math.acos(self.dot(other) / (self.magnitude() * other.magnitude()))


class Line3D:
    def __init__(self, a: Point3D, b: Point3D):
        if a < b:
            self.a: Point3D = a
            self.b: Point3D = b
        else:
            self.a: Point3D = b
            self.b: Point3D = a
        return

    def __repr__(self):
        return f"({self.a}, {self.b})"

    def __eq__(self, other: 'Line3D'):
        return (self.a == other.a and self.b == other.b) or (self.a == other.b and self.b == other.a)

    def __hash__(self):
        return hash((self.a, self.b))

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
    def __init__(self, points: list[Point3D], lines: list[tuple[int, int]] = None):
        """points must be in the winding order of the polygon.
        The winding order is clock wise when looking from the visible face"""
        if lines is None:
            self.lines = self.generate_lines(points)
        else:
            self.lines: list[Line3D] = self.convert_lines(points, lines)

        if not self.validate(points, lines):
            raise InvalidPolygonError
        self.points: list[Point3D] = points

        return

    def __repr__(self):
        return f"{self.points}"

    @staticmethod
    def convert_lines(points, point_pairs) -> list[Line3D]:
        lines = []
        for t in point_pairs:
            lines.append(Line3D(points[t[0]], points[t[1]]))

        return lines

    @staticmethod
    def validate(points: list[Point3D], lines) -> bool:
        """checks if the polygon is valid"""
        if len(points) < 3:
            return False

        if len(set(points)) != len(points):
            return False

        if lines is None:
            return True

        if len(points) != len(lines):
            return False

        point_count = [0] * len(points)
        for t in lines:
            point_count[t[0]] += 1
            point_count[t[1]] += 1

        for i in point_count:
            if i != 2:
                return False

        normal = Vector.from_points(points[1], points[0]).cross(Vector.from_points(points[1], points[2]))
        for point in points[3:]:
            if fp_equals(normal.dot(point), normal.dot(points[0])):
                return False
        return True

    @staticmethod
    def generate_lines(points) -> list[Line3D]:
        """generates the lines from the points"""
        lines = []
        for i in range(len(points)):
            lines.append(Line3D(points[i], points[(i + 1) % len(points)]))

        return lines

    def normal(self) -> Vector:
        """returns the normal vector of the polygon"""
        v1 = Vector.from_points(self.points[1], self.points[0])
        v2 = Vector.from_points(self.points[1], self.points[2])
        return v1.cross(v2)

    def pixels(self, camera: 'Camera') -> set[Point2D]:
        """Returns a set of all pixels that make up the polygon's edges"""
        pixels = set()
        for line in self.project(camera):
            l_culled = camera.projector.screen.line_culling(line)
            if l_culled is None:
                continue
            pixels.update(l_culled.pixels())

        return pixels

    def project(self, camera: 'Camera') -> set[Line2D]:
        lines = set()
        for line in self.lines:
            try:
                l2d = line.project(camera)
            except BehindCameraError:
                continue
            lines.add(l2d)

        return lines

    def facing_camera(self, camera: 'Camera') -> float:
        """checks if the polygon is facing the camera"""
        normal = self.normal()
        camera_vector = Vector.from_points(self.points[0], camera.position)
        return normal.dot(camera_vector)

    def render(self, camera: 'Camera'):
        """Projects and draws the polygon on the screen"""
        if self.facing_camera(camera) > 0:
            return
        pixels = self.pixels(camera)
        camera.projector.screen.draw_pixels(pixels)
        return


class Triangle(Polygon):
    def __init__(self, points: list[Point3D]):
        super().__init__(points)
        return

    def pixels(self, camera: 'Camera') -> set[Point2D]:     # TODO change the implementation to draw the face and not just edges
        """Returns a set of all pixels that make up the triangle's edges"""
        pixels = set()
        for line in self.project(camera):
            l_culled = camera.projector.screen.line_culling(line)
            if l_culled is None:
                continue
            pixels.update(l_culled.pixels())

        return pixels

    def project(self, camera: 'Camera') -> set[Line2D]:
        lines = set()
        for line in self.lines:
            try:
                l2d = line.project(camera)
            except BehindCameraError:
                continue
            lines.add(l2d)

        return lines

    def render(self, camera: 'Camera'):
        """Projects and draws the triangle on the screen"""
        if self.facing_camera(camera) > 0:
            return
        pixels = self.pixels(camera)
        camera.projector.screen.draw_pixels(pixels)
        return


class Quad(Polygon):
    def __init__(self, points: list[Point3D]):
        super().__init__(points)
        return


class SolidOld:
    def __init__(self, points: list[Point3D], lines: list[tuple[int, int]]):
        self.points: list[Point3D] = points
        self.lines: list[Line3D] = self.lines(points, lines)
        return

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

    def project(self, camera: 'Camera') -> set[Line2D]:
        lines = set()
        for line in self.lines:
            try:
                l2d = line.project(camera)
            except BehindCameraError:
                continue
            lines.add(l2d)

        return lines

    def render(self, camera: 'Camera'):
        """Projects and draws the polygon on the screen"""
        pixels = self.pixels(camera)
        camera.projector.screen.draw_pixels(pixels)
        return


class Mesh:
    def __init__(self, polygons: list[Polygon]):
        self.polygons = polygons
        return

    def __repr__(self):
        return f"{self.polygons}"

    def lines(self, camera: 'Camera') -> set[Line3D]:
        """Returns a set of all lines that make up the solid's edges"""
        lines = set()
        for polygon in self.polygons:
            if polygon.facing_camera(camera) < 0:
                lines.update(polygon.lines)

        return lines

    def pixels(self, camera: 'Camera') -> set[Point2D]:
        """Returns a set of all pixels that make up the polygon's edges"""
        pixels = set()
        for line in self.project(camera):
            pixels.update(line.pixels())

        return pixels

    def project(self, camera: 'Camera') -> set[Line2D]:
        lines = set()
        for line in self.lines(camera):
            try:
                l2d = line.project(camera)
            except BehindCameraError:
                continue

            l_culled = camera.projector.screen.line_culling(l2d)
            if l_culled is None:
                continue
            lines.add(l_culled)
        return lines

    def render(self, camera: 'Camera'):
        """Projects and draws the solid on the screen"""
        pixels = self.pixels(camera)
        camera.projector.screen.draw_pixels(pixels)
        return


class Solid(Mesh):
    pass


class Circle(Mesh):
    def __init__(self, center: Point3D, radius: float, normal: Vector, resolution: int = 8):
        if radius < 0:
            raise InvalidPolygonError("Circle radius must be greater than 0")

        super().__init__(Circle.generate_polygons(center, radius, normal, resolution))
        self.center = center
        self.radius = radius
        self.normal = normal
        self.resolution = resolution
        return

    @staticmethod
    def generate_polygons(center: Point3D, radius: float, normal:  Vector, resolution: int = 8) -> list[Polygon]:
        points = circle(radius, center.color, resolution)

        new_points = change_plane(points, normal, center)

        polygons = create_polygons([center], new_points)
        return polygons


class Sphere(Solid):
    def __init__(self, center: Point3D, radius: float, resolution: int = 8):
        super().__init__(Sphere.generate_polygons(center, radius, resolution))
        return

    @staticmethod
    def generate_polygons(center: Point3D, radius: float, resolution: int = 8) -> list[Polygon]:
        assert resolution > 0, "resolution must be greater than 0"
        side_num = 4 * resolution
        angle_step = 2 * math.pi / side_num

        polygons = []

        front = Point3D(center.x, center.y, center.z + radius, center.color)

        points = [front]

        angle = angle_step
        i = 0
        while i < side_num:
            sub_radius = math.sin(angle) * radius
            z = math.cos(angle) * radius

            sub_points = circle(sub_radius, center.color, resolution)
            sub_points = change_plane(sub_points, Vector(0, 0, 1), Point3D(center.x, center.y, center.z + z))
            polygons += create_polygons(points, sub_points)

            points = sub_points
            angle += angle_step
            i += 2
            continue

        return polygons


class Cylinder(Solid):
    def __init__(self, base: Point3D, top: Point3D, base_radius: float, top_radius: float, resolution: int = 1):
        super().__init__(Cylinder.generate_polygons(base, top, base_radius, top_radius, resolution))
        return

    @staticmethod
    def generate_polygons(base: Point3D, top: Point3D, base_r: float, top_r: float, resolution: int) -> list[Polygon]:
        polygons = []
        normal = Vector.from_points(base, top)

        top_circle = circle(top_r, top.color, resolution, cw=True)
        top_points = change_plane(top_circle, normal, top)

        bottom_circle = circle(base_r, base.color, resolution, cw=True)
        bottom_points = change_plane(bottom_circle, -normal, base)

        polygons += create_polygons([top], top_points)
        polygons += create_polygons(top_points, bottom_points, opposite_ways=True)
        polygons += create_polygons([base], bottom_points)

        return polygons


class TruncatedCylinder(Solid):
    def __init__(self, top: Circle, bottom: Circle):
        super().__init__(TruncatedCylinder.generate_polygons(top, bottom))
        return

    @staticmethod
    def generate_polygons(top: Circle, bottom: Circle) -> list[Polygon]:
        polygons = []
        polygons += top.polygons
        polygons += bottom.polygons

        top_points = circle(top.radius, top.center.color, top.resolution)
        top_points = change_plane(top_points, top.normal, top.center)

        bottom_points = circle(bottom.radius, bottom.center.color, bottom.resolution)
        bottom_points = change_plane(bottom_points, bottom.normal, bottom.center)

        polygons += create_polygons(top_points, bottom_points, opposite_ways=True)
        return polygons


class Cone(Cylinder):
    def __init__(self, base: Point3D, top: Point3D, base_radius: float, resolution: int = 1):
        super().__init__(base, top, base_radius, 0, resolution)
        return




# Screen

class Screen:
    def __init__(self, width: int, height: int, pixel_size: int, title: str):
        self.width = width
        self.height = height
        self.pixel_size = pixel_size
        self.title = title
        self.screen = pg.display.set_mode((self.width * self.pixel_size, self.height * self.pixel_size))
        pg.display.set_caption(self.title)
        return

    def update(self):
        pg.display.flip()
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

        if len(intersecting_points) == 0 and not self.in_bounds(line.a) and not self.in_bounds(line.b):
            return None

        assert 0 < len(intersecting_points) <= 2

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
    def __init__(self, projector: 'Projector', position: Point3D = Point3D(0, 0, 0), direction: Vector = Vector(0, 0, 1)):
        self.projector = projector
        self.position: Point3D = position
        self.pitch = 0
        self.yaw = 0
        self.roll = 0
        self.set_direction(direction)

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

    def move(self, direction: Vector):
        self.position += direction
        return

    def rotate_point(self, p: Point3D) -> Point3D:
        # rotation matrix's yaw = y axis, pitch = x axis, roll = y axis
        p = self.rotate_yaw(p)  # yaw needs to go first, but tbh idk why
        p = self.rotate_roll(p)
        p = self.rotate_pitch(p)
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
        x = p.x * self.cos_yaw + p.z * self.sin_yaw
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

        self.pitch = -math.asin(direction_vector.y)

        self.roll = math.atan2(
            direction_vector.x * math.sin(self.pitch) + direction_vector.y * math.cos(self.pitch) - direction_vector.z * math.sin(self.pitch) * math.sin(self.yaw),
            direction_vector.x * math.cos(self.yaw) + direction_vector.z * math.sin(self.yaw)
        )

        self.update_lut()
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



# Settings

class Settings:
    def __init__(self, screen_width: int = 100, screen_height: int = 100, pixel_size: int = 8, fps: int = 10,
                 fov: float = math.pi / 2, pixels_per_unit: int = 555, start_camera_pos: Point3D = Point3D(0, 0, 0),
                 start_camera_dir: Vector = Vector(0, 0, 1), mouse_sensitivity: float = 0.1,
                 scroll_sensitivity: float = 0.1, zoom_sensitivity: float = 0.1):
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.pixel_size: int = pixel_size
        self.fps: int = fps
        self.fov: float = fov
        self.pixels_per_unit: int = pixels_per_unit
        self.start_camera_pos: Point3D = start_camera_pos
        self.start_camera_dir: Vector = start_camera_dir
        self.mouse_sensitivity: float = mouse_sensitivity
        self.scroll_sensitivity: float = scroll_sensitivity
        self.zoom_sensitivity: float = zoom_sensitivity
        return



# Code


def get_rgb_cube() -> Mesh:
    a = Point3D(-1, -1, 2, Color(255, 0, 0))
    b = Point3D(-1, -1, 4, Color(0, 255, 0))
    c = Point3D(1, -1, 2, Color(0, 0, 255))
    d = Point3D(1, -1, 4, Color(0, 255, 0))
    e = Point3D(-1, 1, 2, Color(255, 0, 0))
    f = Point3D(-1, 1, 4, Color(0, 0, 255))
    g = Point3D(1, 1, 2, Color(0, 255, 0))
    h = Point3D(1, 1, 4, Color(255, 0, 0))

    front = Polygon([e, g, c, a])
    back = Polygon([b, d, h, f])
    left = Polygon([a, b, f, e])
    right = Polygon([g, h, d, c])
    top = Polygon([e, f, h, g])
    bottom = Polygon([c, d, b, a])
    cube = Mesh([front, back, left, right, top, bottom])
    return cube


def debug_triangle() -> Polygon:
    a = Point3D(-1, -1, 2, Color(255, 0, 0))
    b = Point3D(-1, -1, 4, Color(0, 255, 0))
    c = Point3D(1, -1, 2, Color(0, 0, 255))
    d = Point3D(1, -1, 4, Color(0, 255, 0))

    connections = [(0, 1), (0, 2), (1, 3), (2, 3)]

    quad = Polygon([a, b, c, d], connections)
    triangle = Polygon([a, b, c], [(0, 1), (0, 2), (1, 2)])
    return triangle


def render_scene(camera: Camera, scene: list[Mesh]):
    for mesh in scene:
        mesh.render(camera)
    return


def motion(config: Settings, camera: Camera, dt: float):
    screen_center = [(config.screen_width * config.pixel_size) >> 1, (config.screen_height * config.pixel_size) >> 1]
    ctrl_pressed = False
    pitch = 0
    yaw = 0
    roll = 0
    movement = Vector(0, 0, 0)

    keys = pg.key.get_pressed()
    if keys[pg.K_LCTRL] or keys[pg.K_RCTRL]:
        ctrl_pressed = True
    if keys[pg.K_w]:
        movement.z += 1
    if keys[pg.K_a]:
        movement.x -= 1
    if keys[pg.K_s]:
        movement.z -= 1
    if keys[pg.K_d]:
        movement.x += 1

    for e in pg.event.get():
        if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
            pg.quit()
            return True

        elif e.type == pg.MOUSEMOTION:
            relative_pos = pg.mouse.get_rel()
            pitch += (-relative_pos[1] / config.pixels_per_unit) * config.mouse_sensitivity
            yaw += (-relative_pos[0] / config.pixels_per_unit) * config.mouse_sensitivity
            pygame.mouse.set_pos(screen_center)
            buttons = pg.mouse.get_pressed()
            if buttons[0]:  # TODO: Add mouse button functionality
                pass
            elif buttons[1]:
                pass
            elif buttons[2]:
                pass

        elif e.type == pg.MOUSEWHEEL:
            if ctrl_pressed and isinstance(camera.projector, PerspectiveProjector):
                camera.projector.fov += e.y * config.zoom_sensitivity
                camera.projector.focal_length = camera.projector.get_focal_length(fov=camera.projector.fov)
            else:
                roll = (-e.y if e.flipped else e.y) * config.scroll_sensitivity

    camera.rotate_pov(pitch, yaw, roll)
    movement = movement.rotate_yaw(camera.yaw)

    # up down doesn't depend on camera rotation
    if keys[pg.K_SPACE]:
        movement.y += 1
    if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]:
        movement.y -= 1

    movement *= dt
    camera.move(movement)

    return False


def main_loop(config: Settings, screen: Screen):
    projector = PerspectiveProjector(screen, config.pixels_per_unit, config.fov)
    camera = Camera(projector, config.start_camera_pos, config.start_camera_dir)

    mesh = []
    mesh.append(load_object_file('cube2.obj'))
    # mesh.append(load_object_file('teapot.obj'))
    # mesh.append(get_rgb_cube())
    # mesh.append(debug_triangle())
    # mesh.append(Sphere(Point3D(0, 0, 2, Color.from_hex("CFB997")), 0.5, 8))
    # mesh.append(Cylinder(Point3D(0, 0, 2), Point3D(0, 3, 2), 0.5, 1, 4))
    # mesh.append(TruncatedCylinder(Circle(Point3D(0, 0, 2), 0.2, Vector(0, -1, 1)), Circle(Point3D(0, 3, 2), 1, Vector(0, 1, 1))))
    # mesh.append(Cone(Point3D(0, 0, 2), Point3D(0, 3, 2), 1, 8))
    # mesh.append(Circle(Point3D(0, 0, 2), 1, Vector(0, 0, -1), 2))

    dt = 1 / config.fps
    clock = pg.time.Clock()
    while True:
        if motion(config, camera, dt):
            break

        render_scene(camera, mesh)
        # debug_triangle()

        screen.update()
        dt = clock.tick(config.fps) / 1000
        print(f'fps = {round(1 / dt, 2)}')
    return


def main():
    screen_width = 200  # in pixels
    screen_height = 200  # in pixels
    pixel_size = 4
    fps = 15
    fov = math.pi / 2
    pixels_per_unit = 555
    start_camera_pos = Point3D(0, 0, 0)
    start_camera_dir = Vector(0, 0, 1)
    mouse_sensitivity = 2
    scroll_sensitivity = 0.1
    zoom_sensitivity = 0.1

    config = Settings(screen_width, screen_height, pixel_size, fps, fov, pixels_per_unit, start_camera_pos,
                        start_camera_dir, mouse_sensitivity, scroll_sensitivity, zoom_sensitivity)

    # check to see if screen gets too big
    if config.screen_width * pixel_size > 1920 or config.screen_height * pixel_size > 1080:
        raise ScreenSizeError('Screen size too big')

    screen = Screen(config.screen_width, config.screen_height, config.pixel_size, '3D Renderer')
    pg.mouse.set_visible(False)

    main_loop(config, screen)

    return None


if __name__ == '__main__':
    main()
