# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
import sys, pygame
import math, random
import os, shutil
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

# IMPORT OBJECT LOADER
from objloader_adam import *

pygame.init()
viewport = (800,600)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
glEnable(GL_LIGHT0)
glEnable(GL_LIGHTING)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded

# LOAD OBJECT AFTER PYGAME INIT
obj = OBJ(sys.argv[1], swapyz=True)

clock = pygame.time.Clock()

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
gluPerspective(90.0, width/float(height), 1, 100.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

'''###trial
glLoadIdentity()
gluLookAt(0.0, 7.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#glTranslate(0.0, 0.0, - 2)
#glRotate(ry, 1, 0, 0)
#glRotate(rx, 0, 1, 0)
glCallList(obj.gl_list)
#pygame.image.save(srf, 'output.png')
pygame.display.flip()
###end of trial'''


# Save
sample_img = {}
counter = 0
mode = 'train'
if mode == 'train':
    iter_time = 10
elif mode == 'valid':
    iter_time = 1
for j in range(iter_time):
    root_path = '/data/' + mode + '_cam_regression'
    directory = root_path + '/0%d' % j
    move_path = root_path + '/move'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(move_path):
        os.makedirs(move_path)
    for i in range(10000): # make 10000 images
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        #gluLookAt(0.0, i, i+1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        rad = 1.742887
        unit_real = 996.679647
        ## CAMERA location
        upb = rad + 15 / unit_real
        lwb = rad + 1 / unit_real
        p_phi_class = random.randint(0, 9)
        p_theta_class = random.randint(0, 9)
        gamma = random.uniform(lwb, upb)
        p_fi = 2 * math.pi * random.uniform(p_phi_class*36, (p_phi_class+1)*36) / 360 # random in every 10 sections
        p_theta = math.pi * random.uniform(p_theta_class*18, (p_theta_class+1)*18) / 180 # random in every 10 sections
        #p_fi = 2 * math.pi * random.uniform(0, 1)
        #p_theta = math.pi * random.uniform(0, 1)
        p_x = gamma * math.sin(p_theta) * math.cos(p_fi)
        p_y = gamma * math.sin(p_theta) * math.sin(p_fi)
        p_z = gamma * math.cos(p_theta)

        ## WHERE does camera look at
        # e_fi = 2 * math.pi * random.uniform(0, 1)
        # e_theta = math.acos(1-2*random.uniform(0, 1))
        # sampled_r = random.uniform(0, rad)
        # fixed look at
        e_fi = 0
        e_theta = 0
        sampled_r = 0
        e_x = sampled_r * math.sin(e_theta) * math.cos(e_fi)
        e_y = sampled_r * math.sin(e_theta) * math.sin(e_fi)
        e_z = sampled_r * math.cos(e_theta)

        ## DIRECTION of camera
        forward = [0,0,0]
        up = [0,0,0]
        #side = [0,0,0]
        forward[0] = e_x - p_x
        forward[1] = e_y - p_y
        forward[2] = e_z - p_z

        # up[0] = random.uniform(0,1)
        # up[1] = random.uniform(0,1)
        # up[2] = random.uniform(0,1)
        up[0] = 0
        up[1] = 0
        up[2] = 1

        norm_forward = normalize(forward)
        side = normalize(crossf(norm_forward, up))
        up = normalize(crossf(side, norm_forward))

        u_x = up[0]
        u_y = up[1]
        u_z = up[2]


        ## SAVE object
        f_name = mode + '_cam_regression{}.png'.format(j*10000+i)
        # sample_img[f_name] = [gamma, p_fi, p_theta, sampled_r, e_fi, e_theta, u_x, u_y, u_z]
        sample_img[f_name] = [gamma, p_fi, p_theta]

        gluLookAt(p_x,p_y,p_z,e_x,e_y,e_z,u_x,u_y,u_z)

        glCallList(obj.gl_list)
        pygame.image.save(srf, directory + '/%s' % f_name)
    archive_name = os.path.expanduser(directory)
    root_dir = os.path.expanduser(directory)
    shutil.make_archive(archive_name, 'gztar', root_dir)
    save_obj(sample_img, root_path + '/gt0%d.pkl' % j)
    _ = shutil.move(archive_name + '.tar.gz', move_path)
    _ = shutil.move(root_path + '/gt0%d.pkl' % j, move_path)

'''rx, ry = (0,0)
tx, ty = (0,0)
zpos = 5
rotate = move = False

glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
glLoadIdentity()

# RENDER OBJECT
glTranslate(tx/20., ty/20., - zpos)
glRotate(ry, 1, 0, 0)
glRotate(rx, 0, 1, 0)
glCallList(obj.gl_list)'''

'''while 1:
    pygame.display.flip()
    pixel_matrix = pygame.surfarray.array2d(display)
    print(pixel_matrix)'''

##
#grap(128,128)
'''while 1:
    clock.tick(30)
    for e in pygame.event.get():
        if e.type == QUIT:
            sys.exit()
        elif e.type == KEYDOWN and e.key == K_ESCAPE:
            sys.exit()
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 4: zpos = max(1, zpos-1)
            elif e.button == 5: zpos += 1
            elif e.button == 1: rotate = True
            elif e.button == 3: move = True
        elif e.type == MOUSEBUTTONUP:
            if e.button == 1: rotate = False
            elif e.button == 3: move = False
        elif e.type == MOUSEMOTION:
            i, j = e.rel
            if rotate:
                rx += i
                ry += j
            if move:
                tx += i
                ty -= j

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # RENDER OBJECT
    glTranslate(tx/20., ty/20., - zpos)
    glRotate(ry, 1, 0, 0)
    glRotate(rx, 0, 1, 0)
    glCallList(obj.gl_list)

    pygame.display.flip()
'''
