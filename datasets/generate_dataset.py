import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import random as rnd
import math
np.set_printoptions(linewidth=400)
img_width = 100
img_height = 100
margins = 2
img = np.zeros((img_width,img_height))

img_rgb = np.zeros((img_width,img_height,3))


problem_points = []


INFINITY = float('inf')
PI = math.pi
cos = math.cos
sin = math.sin
DIRECTION = {'north':(-1,0), 'south':(1,0), 'east':(0,1), 'west':(0,-1),
              'northeast':(-1,1), 'northwest':(-1,-1), 'southeast':(1,1), 'southwest':(1,-1)}


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
    

def generate_points(num_points, max_radius, min_radius, center_coords):
    coords = []
    mean = (max_radius + min_radius)/2.
    std_dev = (max_radius - min_radius)/2.

    #make sure point axis of rotation is inside the image
    center_coords = (clip(center_coords[0],img_height-1,0), clip(center_coords[1],img_width-1,0))

    #iterate thru 360 degrees (2PI radians) and get corresponding cartesian coordinates
    for i in range(num_points):
        radius = clip(rnd.gauss(mean, std_dev), max_radius, min_radius)
        theta = (2*PI*i)/num_points
        x = int(round(center_coords[0] + radius*cos(theta)))
        y = int(round(center_coords[1] + radius*sin(theta)))

        x,y = clip(x, img_height-margins, margins), clip(y, img_width-margins,margins)
        coords.append((x,y))

    #add first point to the end to complete cycle       
    coords.append(coords[0])
    return coords

'''
ALGORITHM FOR DRAWING THE CURVE:
init a cache
examine a = abs(x-x1) and b = abs(y-y1)
if a > b -->map 'first'  to x, or index 0, 'second'  to y, or index 1, and vice versa
inc = min(a,b)/max(a,b)
we start with variables first == second == 0
until either x == x1 or y == y1:
    raise first by 1 and second by inc
    if first >= 1:
        move point (x,y) by 1 in the proper direction along the map['first'] axis
        add the resulting new point (x',y') to the cache
        first -= 1
    if second >= 1:
        same deal but replace first w second

while either x != x1 or y != y1:
    increment until current point = target point

'''
def generate_curve(points):
    curve = []
    
    if len(points) <= 1:
        return curve

    #iterate thru pairs of points to be connected
    for i in range(1,len(points)):
        current = points[i-1]
        target = points[i]
        order_dict = {}
        iteration = []
        
        dx = float(abs(current[0]-target[0]))
        dy = float(abs(current[1]-target[1]))


        if (dx == 0 and dy == 0):
            inc = 0
        else:
            inc = min(dx,dy)/max(dx,dy)
            
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
            if abs(current[0]-target[0]) < abs(current[1]-target[1]) and current[1]-target[1] !=0:
                if current[1] > target[1]:
                    current = move(current,DIRECTION['west'])
                else:
                    current = move(current,DIRECTION['east'])
            else:
                if current[0] > target[0]:
                    current = move(current,DIRECTION['north'])
                else:
                    current = move(current,DIRECTION['south'])

            #if boundary crosses itself, remove a point and try again (still needs tweaking)
            if current in curve[:-1]:
                if i in iteration:
                    print('overlap',i)
                    problem_points.append(i)
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

#example
start = (int(img_width/2),int(img_height/2))       

points = generate_points(30,60,5,start)
curve = generate_curve(points)
print(problem_points)
print(len(problem_points), 'out of', len(points))

for i in range(len(curve)):
    c = curve[i]
    x = c[0]
    y = c[1]
    
    img[x][y] = 1.

    img_rgb[x,y,0] = 1.
    img_rgb[x,y,1] = 1.
    img_rgb[x,y,2] = 1.

for i in range(len(points)):
    p = points[i]
    x = p[0]
    y = p[1]
    if i in problem_points:       
        img[x][y] = 1.

        img_rgb[x,y,0] = 1.
        img_rgb[x,y,1] = 0
        img_rgb[x,y,2] = 0
    else:
        img[p[0]][p[1]] = 1.

        img_rgb[x,y,0] = 0.
        img_rgb[x,y,1] = 1.
        img_rgb[x,y,2] = 0.

for i in range(len(problem_points)):
    p = points[i]
    x = p[0]
    y = p[1]

    if p not in points:
        img_rgb[x,y,0] = 0.
        img_rgb[x,y,1] = 0
        img_rgb[x,y,2] = 1.
    

print(img_rgb.shape)

plt.imshow(img_rgb)
plt.show()

# example points
##[(94, 50), (80, 56), (99, 74), (54, 53), (79, 82),
## (74, 92), (66, 98), (52, 66), (44, 99), (44, 68),
## (45, 56), (46, 53), (12, 67), (31, 54), (1, 15),
## (39, 38), (20, 0), (31, 0), (44, 0), (55, 0), (61, 17),
## (57, 38), (53, 46), (61, 42), (99, 27), (94, 50)]

# test for errors
##data = []
##for i in range(1000):
##    print i
##    points = generate_points(30,60,5,start)
##    curve = generate_curve(points)
##
##print(problem_points)
