import cv2
import pygame
import pickle
import numpy as np
from OpenGL.GL import *


def MTL(filename):
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise (ValueError, "mtl file doesn't start with newmtl stmt")
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
            surf = pygame.image.load(mtl['map_Kd'])
            image = pygame.image.tostring(surf, 'RGBA', 1)
            ix, iy = surf.get_rect().size
            texid = mtl['texture_Kd'] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                            GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                            GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, image)
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents


class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = float(values[1]), float(values[2]), float(values[3])
                if swapyz:
                    v = float(values[1]), float(values[3]), float(values[2])
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = float(values[1]), float(values[2]), float(values[3])
                if swapyz:
                    v = float(values[1]), float(values[3]), float(values[2])
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            mtl = self.mtl[material]
            if 'texture_Kd' in mtl:
                # use diffuse texmap
                glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
            else:
                # just use diffuse colour
                glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()


def grap(w, h):
    data = []
    glReadBuffer(GL_FRONT)
    data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.zeros((h * w * 3), dtype=np.uint8)
    for i in range(0, len(data), 3):
        arr[i] = data[i + 2]
        arr[i + 1] = data[i + 1]
        arr[i + 2] = data[i]
    arr = np.reshape(arr, (h, w, 3))

    cv2.flip(arr, 0, arr)
    cv2.putText(arr, "Opencv", (40, 40), cv2.FONT_ITALIC, 1, (0, 255, 0))
    cv2.putText(arr, "Miha_Singh", (40, 70), cv2.FONT_ITALIC, 1, (0, 255, 0))
    cv2.imshow('scene', arr)
    cv2.waitKey(1)


def save_obj(obj, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)


def normalize(coord):
    temp = [0, 0, 0]
    temp[0] = coord[0]
    temp[1] = coord[1]
    temp[2] = coord[2]
    l = (temp[0] ** 2 + temp[1] ** 2 + temp[2] ** 2) ** 0.5
    temp[0] /= l
    temp[1] /= l
    temp[2] /= l
    return temp


def crossf(a, b):
    temp = [0, 0, 0]
    temp[0] = a[1] * b[2] - a[2] * b[1]
    temp[1] = a[2] * b[0] - a[0] * b[2]
    temp[2] = a[0] * b[1] - a[1] * b[0]
    return temp
