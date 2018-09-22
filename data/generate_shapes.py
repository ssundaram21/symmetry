import numpy as np
#import matplotlib
#from matplotlib import pyplot as plt
import random as rnd
import math
import sys

import floodfill

sys.setrecursionlimit(3000)
np.set_printoptions(linewidth=400)
margins = 2

INFINITY = float('inf')
PI = math.pi
cos = math.cos
sin = math.sin
DIRECTION = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1),
             'northeast': (-1, 1), 'northwest': (-1, -1), 'southeast': (1, 1), 'southwest': (1, -1)}

#test

#return new coordinates of a point when moved one unit in direction 'direction'
def move(start_position, direction):
    return (start_position[0] + direction[0], start_position[1] + direction[1])


def new_position(start_position, index, direction_sign):
    if direction_sign == 0:
        return start_position
    
    if index == 0:
        if direction_sign > 0:
            direction = DIRECTION['south']
        else:
            direction = DIRECTION['north']           
    else:
        if direction_sign > 0:
            direction = DIRECTION['east']
        else:
            direction = DIRECTION['west']

    return (start_position[0] + direction[0], start_position[1] + direction[1])


#make sure a given number stays within upper_bound and lower_bound
def clip(number, upper_bound=INFINITY, lower_bound=-INFINITY):
    if number < lower_bound:
        return int(lower_bound)
    elif number > upper_bound:
        return int(upper_bound)
    else:
        return int(number)


def sign(number):
    if number < 0:
        return -1
    elif number > 0:
        return 1
    return 0
    

def generate_points(num_points, image_width, image_height, max_radius, min_radius, center_coords):
    coords = []
    mean = (max_radius + min_radius)/2.
    std_dev = (max_radius - min_radius)/2.

    #make sure point axis of rotation is inside the image
    center_coords = (clip(center_coords[0], image_height-margins, 0), clip(center_coords[1], image_width-margins, 0))

    #iterate thru 360 degrees (2PI radians) and get corresponding cartesian coordinates
    for i in range(num_points):
        radius = clip(rnd.gauss(mean, std_dev), max_radius, min_radius)
        theta = (2*PI*i)/num_points
        x = int(round(center_coords[0] + radius*cos(theta)))
        y = int(round(center_coords[1] + radius*sin(theta)))

        x, y = clip(x, image_height-margins, margins), clip(y, image_width-margins,margins)
        coords.append((x, y))

    #add first point to the end to complete cycle       
    coords.append(coords[0])
    return coords


def generate_curve(points):
    curve = []
    
    if len(points) <= 1:
        return curve

    #iterate thru pairs of points to be connected
    for i in range(1, len(points)):
        current = points[i-1]
        target = points[i]
        order_dict = {}
        iteration = []
        
        dx = float(abs(current[0]-target[0]))
        dy = float(abs(current[1]-target[1]))

        if (dx == 0 and dy == 0):
            inc = 0
        else:
            inc = min(dx, dy)/max(dx, dy)
            
        first_value = 0 
        second_value = 0

        sign_xy = (sign(target[0]-current[0]), sign(target[1]-current[1]))
        
        if dx > dy:
            first_index = 0
            second_index = 1
        else:
            first_index = 1
            second_index = 0
        
        while current[0] != target[0] and current[1] != target[1]:
            first_value += 1
            second_value += inc
            
            if first_value >= 1:
                first_value -= 1
                current = new_position(current, first_index, sign_xy[first_index])
                curve.append(current)
                
            if second_value >= 1:
                second_value -= 1
                current = new_position(current, second_index, sign_xy[second_index])
                curve.append(current)

        #connect pair of points (tweak)
        while current != target:
            if abs(current[0]-target[0]) < abs(current[1]-target[1]) and current[1]-target[1] != 0:
                if current[1] > target[1]:
                    current = move(current, DIRECTION['west'])
                else:
                    current = move(current, DIRECTION['east'])
            else:
                if current[0] > target[0]:
                    current = move(current, DIRECTION['north'])
                else:
                    current = move(current, DIRECTION['south'])

            #if boundary crosses itself, remove a point and try again (still needs tweaking)
            if current in curve[:-1]:
                if i in iteration:
                    curve = []
                    if target != points[0]:
                        points.remove(target)
                    else:
                        points.remove(points[i-1])#points = generate_points(4,int(img_height/3),int(img_height/4),(int(img_height/2),int(img_width/2)))
                    curve = generate_curve(points)
                    return curve
                    
                iteration.append(i)               
            curve.append(current)
    return curve


def visit(image, position, visited, not_visited, directions, depth=0):
    move_dict = {}
    move_dict['north'] = move(position, DIRECTION['north'])
    move_dict['south'] = move(position, DIRECTION['south'])
    move_dict['east'] = move(position, DIRECTION['east'])
    move_dict['west'] = move(position, DIRECTION['west'])
    move_dict['northwest'] = move(position, DIRECTION['northwest'])
    move_dict['northeast'] = move(position, DIRECTION['northeast'])
    move_dict['southwest'] = move(position, DIRECTION['southwest'])
    move_dict['southeast'] = move(position, DIRECTION['southeast'])

    if position in visited:
        return

    if depth > 2000:
        not_visited[position] = True
        return
    
    visited[position] = True
    if position in not_visited:
        del not_visited[position]

    if image[position] == 1.:
        visited[position] = False
        return

    else:
        for d in directions:
            visit(image, move_dict[d], visited, not_visited, directions, depth + 1)
        return
    

def find_ground_truth(image, center_coords):
    visited = {}
    not_visited = {}
    visit(image, center_coords, visited, not_visited, ['north', 'south', 'east', 'west'])
    for point in visited:
        image[point] = 1.
    

def apply_mask(img_filled, img_mask):
    img = np.zeros((img_filled.shape[0], img_filled.shape[1]), dtype=np.uint8)
    for x in range(1, img_filled.shape[0]-1):
        for y in range(1, img_filled.shape[1]-1):
            if img_filled[x][y]:
                if img_mask[x + 1][y]:
                    img[x + 1][y] = 1
                if img_mask[x - 1][y]:
                    img[x - 1][y] = 1
                if img_mask[x][y + 1]:
                    img[x][y + 1] = 1
                if img_mask[x][y - 1]:
                    img[x][y - 1] = 1
                if img_mask[x + 1][y + 1]:
                    img[x + 1][y + 1] = 1
                if img_mask[x + 1][y - 1]:
                    img[x + 1][y - 1] = 1
                if img_mask[x - 1][y + 1]:
                    img[x - 1][y + 1] = 1
                if img_mask[x - 1][y - 1]:
                    img[x - 1][y - 1] = 1
    return img

def generate_data(num_points, image_width, image_height, max_radius, min_radius):
    center_coords = (int(image_width/2), int(image_height/2))
    
    img = np.zeros((image_width, image_height))
    
    points = generate_points(num_points, image_width, image_height, max_radius, min_radius, center_coords)
    curve = generate_curve(points)

    for i in range(len(curve)):
        c = curve[i]
        x = c[0]
        y = c[1]

        img[x][y] = 1.

    for i in range(len(points)):
        p = points[i]
        x = p[0]
        y = p[1]

        img[x][y] = 1.

    img = img.astype(np.uint8)
    for x in range(1, img.shape[0]-1):
        for y in range(1, img.shape[1]-1):
            if img[x + 1][y] + img[x - 1][y] + img[x][y + 1] + img[x][y - 1] == 1:
                img[x][y] = 0
            #if img[x + 1][y] + img[x - 1][y] + img[x][y + 1] + img[x][y - 1] == 4:
            #    img[x][y] = 1

    img_filled_original = floodfill.from_edges(img, four_way=True)
    img_filled = (img_filled_original - img_filled_original * img).astype(np.uint8)
    img = apply_mask(img_filled, img)

    ''' 
    from PIL import Image;
    imga = Image.fromarray(128 * img);
    imga.save('testrgbA.png')
    imga = Image.fromarray(128 * img_filled);
    imga.save('testrgbB.png')
    '''

    img_filled = (1 - img_filled_original)
    #img_filled = (img_filled_original - img_filled_original * img).astype(np.uint8)
    img = apply_mask(img_filled, img)

    img_filled_original = floodfill.from_edges(img, four_way=True)
    img_filled_original = (img_filled_original - img_filled_original * img).astype(np.uint8)

    '''
    from PIL import Image;
    imga = Image.fromarray(128 * img);
    imga.save('testrgb1.png')
    imga = Image.fromarray(128 * img_filled_original);
    imga.save('testrgb.png')
    '''

    return img, img_filled_original

##img = generate_data(50, 100, 100, 100, 100)
##find_ground_truth(img, (49,49))
##find_ground_truth(img, (20,20))
##
##plt.imshow(img)
##plt.show()


