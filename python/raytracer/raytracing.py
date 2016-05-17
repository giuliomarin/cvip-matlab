import numpy as np
import matplotlib.pyplot as plt
import os
from random import shuffle
import time
import glob
from multiprocessing import Pool

def cross(a, b):
    return np.asarray([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


def translate(x, y, z):
    M = np.eye(4, dtype = np.float32)
    M[0, 3] = x
    M[1, 3] = y
    M[2, 3] = z
    return M


def rotate(x, y, z, a):
    a = a / 180.0 * np.pi
    R1 = np.cos(a) * np.asmatrix(np.eye(3, dtype = np.float32))
    R2 = (1 - np.cos(a)) * np.asmatrix([[x * x, x * y, x * z], [x * y, y * y, y * z], [x * z, y * z, z * z]])
    R3 = np.sin(a) * np.asmatrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    M = np.asmatrix(np.eye(4, dtype = np.float32))
    M[0:3, 0:3] = R1 + R2 + R3
    return M


def scale(x, y, z):
    M = np.asmatrix(np.eye(4, dtype = np.float32))
    M[0, 0] = x
    M[1, 1] = y
    M[2, 2] = z
    return M


def parsefile(filepath):
    camera = Camera()  # Camera(640 / 4, 480 / 4, np.array([0.5, 0.36, -2.]), np.array([0., 0., 0.]), np.array([0., 1., 0.]), 60.0)

    scene = []  # [Sphere([.75, .1, 1.], 0.6, [0., 0., 1.0]), Plane([0., 0.0, 5.], [0., 0., -1.], [0.0, 0.1, 0.0], [0.0, 1.0, 0.0])]

    light = []  # [Point([5.0, 10.0, -10.0], [1.0, 1.0, 1.0], 1.0, [1.0, 0.0, 0.0])]

    param = Parameters()

    vertex = []
    vertexnormal = []

    transfstack = [np.asmatrix(np.eye(4, dtype = np.float32))]

    # Default global parameters
    attenuation = np.asarray([1.0, 0.0, 0.0])

    # Default parameters per object
    ambient = np.asarray([0.2, 0.2, 0.2])
    diffuse = np.asarray([0.0, 0.0, 0.0])
    specular = np.asarray([0.0, 0.0, 0.0])
    emission = np.asarray([0.0, 0.0, 0.0])
    shininess = 0.0

    fid = open(filepath, 'r')
    for line in fid:
        line = line.lstrip()
        # skip empty lines and comments
        if len(line) == 0 or line.startswith('#'):
            continue
        line = line.split()
        cmd = line[0]

        # Camera parameters
        if cmd == 'size':
            camera.setsize(int(line[1]), int(line[2]))
        elif cmd == 'camera':
            camera.setpose(
                np.asarray([float(line[1]), float(line[2]), float(line[3])]),  # eye
                np.asarray([float(line[4]), float(line[5]), float(line[6])]),  # center
                np.asarray([float(line[7]), float(line[8]), float(line[9])]),  # up
                float(line[10]))  # fov

        # Lights and objects
        elif cmd == 'attenuation':
            attenuation = np.asarray([float(line[1]), float(line[2]), float(line[3])])
        elif cmd == 'directional':
            obj = Directional(np.asarray([float(line[1]), float(line[2]), float(line[3])]),  # direction
                              np.asarray([float(line[4]), float(line[5]), float(line[6])]))  # color
            obj.applytransform(transfstack[-1])
            light.append(obj)
        elif cmd == 'point':
            obj = Point(np.asarray([float(line[1]), float(line[2]), float(line[3])]),  # position
                        np.asarray([float(line[4]), float(line[5]), float(line[6])]),  # color
                        attenuation)  # attenuation
            obj.applytransform(transfstack[-1])
            light.append(obj)
        elif cmd == 'ambient':
            ambient = np.asarray([float(line[1]), float(line[2]), float(line[3])])
        elif cmd == 'diffuse':
            diffuse = np.asarray([float(line[1]), float(line[2]), float(line[3])])
        elif cmd == 'specular':
            specular = np.asarray([float(line[1]), float(line[2]), float(line[3])])
        elif cmd == 'shininess':
            shininess = float(line[1])
        elif cmd == 'emission':
            emission = np.asarray([float(line[1]), float(line[2]), float(line[3])])

        elif cmd == 'maxverts':
            vertex = []
        elif cmd == 'maxvertnormals':
            vertexnormal = []
        elif cmd == 'vertex':
            vertex.append(np.asarray([float(line[1]), float(line[2]), float(line[3])]))
        elif cmd == 'vertexnormal':
            vertex.append(np.asarray(
                [float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])]))

        elif cmd == 'tri':
            obj = Triangle([vertex[i] for i in (int(line[1]), int(line[2]), int(line[3]))],
                           ambient,
                           diffuse,
                           specular,
                           shininess,
                           emission)
            obj.applytransform(transfstack[-1])
            scene.append(obj)
        elif cmd == 'sphere':
            obj = Sphere(np.asarray([float(line[1]), float(line[2]), float(line[3])]),  # center
                         float(line[4]),  # radius
                         ambient,
                         diffuse,
                         specular,
                         shininess,
                         emission)
            obj.applytransform(transfstack[-1])
            scene.append(obj)
        # Transformations
        elif cmd == 'translate':
            transfstack[-1] *= translate(float(line[1]), float(line[2]), float(line[3]))
        elif cmd == 'scale':
            transfstack[-1] *= scale(float(line[1]), float(line[2]), float(line[3]))
        elif cmd == 'rotate':
            transfstack[-1] *= rotate(float(line[1]), float(line[2]), float(line[3]), float(line[4]))

        elif cmd == 'pushTransform':
            transfstack.append(transfstack[-1].copy())
        elif cmd == 'popTransform':
            transfstack.pop()

        # Other parameters
        elif cmd == 'maxdepth':
            param.maxdepth = int(line[1])
        elif cmd == 'output':
            param.outfilename = line[1]

    fid.close()

    return camera, scene, light, param


class Camera:
    def __init__(self, width = None, height = None, eye = None, center = None, up = None, fovy = None):
        self.width = width
        self.height = height
        self.eye = eye
        self.center = center
        self.up = up
        self.fovyd = fovy
        if fovy is not None:
            self.fovyr = fovy / 180.0 * np.pi

    def setsize(self, width, height):
        self.width = width
        self.height = height

    def setpose(self, eye, center, up, fovy):
        self.eye = eye
        self.center = center
        self.up = up
        self.fovyd = fovy
        self.fovyr = fovy / 180.0 * np.pi


class Parameters:
    def __init__(self):
        self.outfilename = 'raytracing.png'
        self.maxdepth = 5


class Light:
    attenuation = np.asarray([1.0, 0.0, 0.0])


class Directional(Light):
    def __init__(self, direction, color, attenuation = np.asarray([1.0, 0.0, 0.0])):
        self.direction = normalize(np.asarray(direction))
        self.color = np.asarray(color)
        self.attenuation = attenuation

    def getdirection(self, p):
        return -self.direction

    def getdistance(self, p):
        return np.inf

    def applytransform(self, M):
        pass


class Point(Light):
    def __init__(self, position, color, attenuation = np.asarray([1.0, 0.0, 0.0])):
        self.position = np.asarray(position)
        self.color = np.asarray(color)
        self.attenuation = attenuation

    def getdirection(self, p):
        return normalize(p - self.position)

    def getdistance(self, p):
        return np.linalg.norm(p - self.position)

    def applytransform(self, M):
        pass


class Object:
    def intersect(self, ray):
        pass

    def getnormal(self, ray):
        pass


class Triangle(Object):
    def __init__(self, vertex, ambient, diffuse = np.array([0.0, 0.0, 0.0]),
                 specular = np.array([0.0, 0.0, 0.0]), shininess = 50.0, emission = np.array([0.0, 0.0, 0.0])):

        # Shape
        self.a = np.asarray(vertex[0])
        self.b = np.asarray(vertex[1])
        self.c = np.asarray(vertex[2])
        self.normal = normalize(cross(self.b - self.a, self.c - self.a))

        # Color
        self.ambient = np.asarray(ambient)
        self.diffuse = np.asarray(diffuse)
        self.emission = np.asarray(emission)
        self.specular = np.asarray(specular)
        self.shininess = shininess

    def getnormal(self, ray):
        return self.normal

    def intersect(self, ray):
        e1 = self.b - self.a
        e2 = self.c - self.a
        P = cross(ray[1], e2)
        det = np.dot(e1, P)
        if abs(det) < 1e-6:
            return np.inf
        inv_det = 1.0 / det
        T = ray[0] - self.a
        u = np.dot(T, P) * inv_det
        if u < 0 or u > 1:
            return np.inf
        Q = cross(T, e1)
        v = np.dot(ray[1], Q) * inv_det
        if v < 0 or u + v > 1:
            return np.inf
        t = np.dot(e2, Q) * inv_det
        if t > 1e-6:
            return t
        return np.inf

    def applytransform(self, M):
        a = np.asmatrix(np.append(self.a, 1)).T
        self.a = np.asarray(np.dot(M, a).T)[0][0:3]
        b = np.asmatrix(np.append(self.b, 1)).T
        self.b = np.asarray(np.dot(M, b).T)[0][0:3]
        c = np.asmatrix(np.append(self.c, 1)).T
        self.c = np.asarray(np.dot(M, c).T)[0][0:3]
        self.normal = normalize(cross(self.b - self.a, self.c - self.a))


class Plane(Object):
    def __init__(self, position, normal, ambient, diffuse = np.array([0.0, 0.0, 0.0]),
                 specular = np.array([0.0, 0.0, 0.0]), shininess = 50.0, emission = np.array([0.0, 0.0, 0.0])):
        # Shape
        self.position = np.asarray(position)
        self.normal = np.asarray(normal)

        # Color
        self.ambient = np.asarray(ambient)
        self.diffuse = np.asarray(diffuse)
        self.emission = np.asarray(emission)
        self.specular = np.asarray(specular)
        self.shininess = shininess

    def intersect(self, ray):
        denom = np.dot(ray[1], self.normal)
        if np.abs(denom) < 1e-6:
            return np.inf
        d = np.dot(self.position - ray[0], self.normal) / denom
        return d if (d > 0) else np.inf

    def getnormal(self, ray):
        return self.normal

    def applytransform(self, M):
        pass


class Checkerboard(Object):
    def __init__(self, position, normal, ambient, diffuse = np.array([0.0, 0.0, 0.0]),
                 specular = np.array([0.0, 0.0, 0.0]), shininess = 50.0, emission = np.array([0.0, 0.0, 0.0])):
        # Shape
        self.position = np.asarray(position)
        self.normal = np.asarray(normal)

        # Color
        self.ambient = np.asarray(ambient)
        self.diffuse = np.asarray(diffuse)
        self.emission = np.asarray(emission)
        self.specular = np.asarray(specular)
        self.shininess = shininess

    def intersect(self, ray):
        denom = np.dot(ray[1], self.normal)
        if np.abs(denom) < 1e-6:
            return np.inf
        d = np.dot(self.position - ray[0], self.normal) / denom
        return d if (d > 0) else np.inf

    def getnormal(self, ray):
        return self.normal

    def applytransform(self, M):
        pass


class Sphere(Object):
    def __init__(self, center, radius, ambient, diffuse = np.array([0.0, 0.0, 0.0]),
                 specular = np.array([0.0, 0.0, 0.0]), shininess = 50.0, emission = np.array([0.0, 0.0, 0.0])):

        # Shape
        self.center = np.asarray(center)
        self.radius = float(radius)

        # Color
        self.ambient = np.asarray(ambient)
        self.diffuse = np.asarray(diffuse)
        self.emission = np.asarray(emission)
        self.specular = np.asarray(specular)
        self.shininess = shininess

        # Transformation
        self.M = np.asmatrix(np.eye(4, dtype = np.float32))
        self.Minv = np.asmatrix(np.eye(4, dtype = np.float32))

    def intersect(self, ray):
        # Apply inverse transform
        ray_dist = list(ray)
        ray0 = np.asmatrix(np.append(ray_dist[0], 1)).T
        ray_dist[0] = np.asarray(np.dot(self.Minv, ray0).T)[0][0:3]
        ray1 = np.asmatrix(np.append(ray_dist[1], 0)).T
        ray_dist[1] = np.asarray(np.dot(self.Minv, ray1).T)[0][0:3]

        # Compute standard ray-surface intersection
        d = np.inf
        a = np.dot(ray_dist[1], ray_dist[1])
        PS = ray_dist[0] - self.center
        b = 2 * np.dot(ray_dist[1], PS)
        c = np.dot(PS, PS) - self.radius ** 2
        disc = b ** 2 - 4 * a * c
        if disc > 0:
            sqdisc = np.sqrt(disc)
            d12 = ((-b + sqdisc) / (2.0 * a), (-b - sqdisc) / (2.0 * a))
            dmin, dmax = min(d12), max(d12)
            if dmax >= 0:
                d = dmax if dmin < 0 else dmin
        if d == np.inf:
            return np.inf

        # Transform back to actual coordinates
        Pdist = np.asmatrix(np.append(ray_dist[0] + d * ray_dist[1], 1)).T
        Preal = np.asarray(np.dot(self.M, Pdist).T)[0][0:3]
        return np.linalg.norm(Preal - ray[0])

    def getnormal(self, ray):
        if len(ray) == 2:
            # is a ray
            d = self.intersect(ray)
            point = ray[0] + d * ray[1]
        else:
            # is a point
            point = np.asarray(ray)
        point = np.asmatrix(np.append(point, 1)).T
        point_dist = np.asarray(np.dot(self.Minv, point).T)[0][0:3]
        normal_dist = np.asmatrix(np.append(normalize(point_dist - self.center), 0)).T
        normal = normalize(np.asarray(np.dot(self.Minv.T, normal_dist).T)[0][0:3])
        return normal

    def applytransform(self, M):
        self.M = M
        self.Minv = np.linalg.inv(M)


def normalize(x):
    x /= np.linalg.norm(x)
    return x


# def get_color(obj, M):
#     color = obj['color']
#     if not hasattr(color, '__len__'):
#         color = color(M)
#     return color
#
# colorplane = lambda M: (color_plane0 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1)


def trace_ray(ray, scene, light):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = obj.intersect(ray)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    P = ray[0] + ray[1] * t
    # Find properties of the object.
    N = obj.getnormal(P)

    c_ray = obj.ambient + obj.emission
    # Shadow: find if the point is shadowed or not.
    for l in light:
        l_ray = (P + N * .0001, -l.getdirection(P))
        r = l.getdistance(P)
        d = [obj_sh.intersect(l_ray) for k, obj_sh in enumerate(scene) if k != obj_idx]
        if d and min(d) < r:
            continue

        # Computing the color.
        H = normalize(l_ray[1] - ray[1])
        if r == np.inf:
            r = 0.0
        c_ray += l.color / (l.attenuation[0] + l.attenuation[1] * r + l.attenuation[2] * r ** 2) * \
                 (obj.diffuse * max(np.dot(N, l_ray[1]), 0) + obj.specular * (max(np.dot(N, H), 0) ** obj.shininess))
    return obj, P, N, c_ray


def printbar(curr, total, size = 20, freq = 10, pre = ''):
    if curr % freq is not 0:
        return
    num_done = int(float(curr) / float(total) * size)
    bar = pre + '['
    for e in xrange(num_done):
        bar += '='
    for e in xrange(size - num_done):
        bar += ' '
    bar += ']'
    print bar + ' %d %%' % (float(curr) / float(total) * 100)


def processfile(filename, idfile = 0):

    if isinstance(filename, tuple):
        idfile = filename[1]
        filename = filename[0]
    print '[' + str(idfile) + '] ' + 'Processing input: %s' % filename

    foldername = os.path.dirname(filename)
    scenedata = parsefile(filename)
    camera, scene, light, param = scenedata

    start_time = time.time()
    p = Pool(NUM_STRIPES)
    results = p.map(processstripe, zip([scenedata]*NUM_STRIPES, [idfile]*NUM_STRIPES, range(NUM_STRIPES)))
    img = sum(results)
    # img = processstripe(3)
    printbar(1, 1, pre = '[' + str(idfile) + '] ')
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.imsave(os.path.join(foldername, param.outfilename), img)
    # plt.imshow(img)
    # plt.show()


# Parallelize calls
def processstripe((scenedata, idfile, idstripe)):

    camera, scene, light, param = scenedata
    img = np.zeros((camera.height, camera.width, 3))

    # Precompute camera coordinate system
    c_w = normalize(camera.eye - camera.center)
    c_u = normalize(cross(camera.up, c_w))
    c_v = cross(c_w, c_u)

    f = 1.0 / np.tan(camera.fovyr / 2.0)
    max_y = 1.0
    r = float(camera.width) / camera.height
    max_x = 1.0 * r

    # Loop through all pixels.
    col = np.zeros(3)  # Current color.

    START_STRIPES = 0
    END_STRIPES = camera.height
    if (END_STRIPES - START_STRIPES) % NUM_STRIPES is not 0:
        raise Exception('Impossible to divide input in %d stripes' % NUM_STRIPES)
    SIZE_STRIPE = (END_STRIPES - START_STRIPES) / NUM_STRIPES
    startstripe = START_STRIPES + idstripe * SIZE_STRIPE
    endstripe = START_STRIPES + (idstripe + 1) * SIZE_STRIPE
    hrange = range(startstripe, endstripe)
    wrange = range(camera.width)
    shuffle(hrange)
    shuffle(wrange)
    for i, y in enumerate(hrange):
        printbar(i, len(hrange), pre = '[' + str(idfile) + '|' + str(idstripe) + '] ')
        for x in wrange:
            # Ray direction
            a = max_x * ((x + 0.5) - camera.width / 2) / (camera.width / 2)
            b = max_y * (camera.height / 2 - (y + 0.5)) / (camera.height / 2)
            direction = normalize(a * c_u + b * c_v - f * c_w)
            ray = (camera.eye, direction)

            # Reset values
            reflection = np.array([1.0, 1.0, 1.0])
            col[:] = 0

            # Loop through initial and secondary rays.
            depth = 0
            while depth < param.maxdepth:
                traced = trace_ray(ray, scene, light)
                if not traced:
                    break
                obj, P, N, col_ray = traced
                # Reflection: create a new ray.
                ray = (P + N * 1e-3, normalize(ray[1] - 2 * np.dot(ray[1], N) * N))
                depth += 1
                col += reflection * col_ray
                reflection *= obj.specular
                if not np.any(reflection):
                    break
            img[y, x, :] = np.clip(col, 0, 1)
    return img

if __name__ == '__main__':

    filetotest_path = '/Users/giulio/Desktop/edX-OpenGL/hw3-submissionscenes/*.test'
    filetotest = glob.glob(filetotest_path) # ['/Users/giulio/Desktop/edX-OpenGL/hw3-submissionscenes/scene4-specular.test']
    filetotestid = zip(filetotest, range(len(filetotest)))

    NUM_STRIPES = 4

    for datatoprocess in filetotestid:
        processfile(datatoprocess)


