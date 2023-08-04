import math
import pygame.draw
from pygame import display, event, QUIT, time


# Objects

class Point:
    def __init__(self, x: int|float, y: int|float, z: int|float = 0):
        self.x: int|float = x
        self.y: int|float = y
        self.z: int|float = z
        return

    @staticmethod
    def from_tuple(t: tuple[int|float, ...]):
        if len(t) == 1:
            return Point(t[0], 0)
        elif len(t) == 2:
            return Point(t[0], t[1])
        elif len(t) >= 3:
            return Point(t[0], t[1], t[2])
        else:
            assert False

    def __add__(self, other: 'Point'):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point'):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: int|float):
        return Point(self.x * other, self.y * other, self.z * other)

    def distance(self, other: 'Point') -> int|float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5

    def manhattan_distance(self, other: 'Point') -> int|float:
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)

    def project(self, projector: 'Projector') -> 'Point':
        return projector.project(self)


class Vector(Point):
    def __init__(self, x, y, z = 0):
        super().__init__(x, y, z)
        return

    def dot(self, other: 'Vector') -> int|float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector') -> 'Vector':
        return Vector(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)


class Line:
    def __init__(self, a: Point, b: Point):
        self.a: Point = a
        self.b: Point = b
        return

    def length(self) -> float:
        return self.a.distance(self.b)

    def manhattan_length(self) -> float:
        return self.a.manhattan_distance(self.b)

    def project(self, projector: 'Projector'):
        return Line(self.a.project(projector), self.b.project(projector))


class Polygon:
    def __init__(self, points: list[Point], lines: list[tuple[int, int]]):
        self.points: list[Point] = points
        self.point_pairs: list[tuple[int, int]] = lines
        return

    def lines(self) -> list[Line]:
        lines = []
        for t in self.point_pairs:
            lines.append(Line(self.points[t[0]], self.points[t[1]]))

        return lines

    def project(self, projector: 'Projector') -> list[Line]:
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

    def in_bounds(self, x: int, y: int, width: int, height: int):
        return 0 <= x < width and 0 <= y < height

    def pixel(self, p: Point, color: tuple[int, int, int]):
        coords = (p.x + (self.width >> 1), (self.height >> 1) - p.y)  # center the screen
        # up is positive y, right is positive x

        if self.in_bounds(coords[0], coords[1], self.width, self.height):
            pixel = pygame.Rect(coords[0] * self.pixel_size, coords[1] * self.pixel_size, self.pixel_size, self.pixel_size)
            pygame.draw.rect(self.screen, color, pixel)
        return

    def line(self, a: Point, b: Point, color: tuple[int, int, int]):
        # modified dda

        if a.x > b.x:
            a, b = b, a

        dx = b.x - a.x
        dy = b.y - a.y

        vertical_line = abs(dx) < abs(dy)

        factor = 0
        while (1 << factor) < abs(dx) or (1 << factor) < abs(dy):
            factor += 1

        half = 1 << (factor - 1)
        endx = (b.x << factor) + half
        endy = (b.y << factor) + half
        x = (a.x << factor) + half
        y = (a.y << factor) + half

        while x != endx or y != endy:
            prev_pixel = Point(x >> factor, y >> factor)
            x += dx
            y += dy
            if (vertical_line and prev_pixel.y != (x >> factor)) or (not vertical_line and prev_pixel.x != (x >> factor)):
                self.pixel(prev_pixel, color)

        self.pixel(Point(x >> factor, y >> factor), color)
        return


# Projectors

class Projector:    # default is orthographic projection
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        return

    def project(self, p: Point) -> Point:
        return p


class PerspectiveProjector(Projector):
    def __init__(self, width: int, height: int, fov: float):
        super().__init__(width, height)
        self.fov = fov
        self.focal_length = self.get_focal_length(fov)
        return

    def get_focal_length(self, fov):
        return (self.width // 2) / math.tan(fov / 2)

    def project(self, p: Point) -> Point:
        x = (p.x * self.focal_length) // (p.z + self.focal_length)
        y = (p.y * self.focal_length) // (p.z + self.focal_length)
        return Point(x, y)


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

        screen.update()

        clock.tick(10)
    return None


if __name__ == '__main__':
    main()
