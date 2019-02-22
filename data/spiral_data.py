#This is really badly written, but the function "create_data_set(x)" 
#returns a dictionary with x keys and such that the values correspond 
#to ordered pairs (A,B), where A is a 30x30 matrix (where the pixels in the path have pixels
#1 and the other pixels have all value 0) and B is the ground truth
#(pixels outside are 0 and pixels inside are 1).
#note: Matrices are represented by lists of lists, i.e. A[x][y] corresponds to the pixel (x,y).


from random import randint
import numpy as np
import operator


VECTORS ={0:(0,-1), 1:(1,-1),2:(1,0),3:(1,1),4:(0,1),5:(-1,1),6:(-1,0),7:(-1,-1)}
INV_VECTORS={(0,-1):0, (1,-1):1,(1,0):2,(1,1):3,(0,1):4,(-1,1):5,(-1,0):6,(-1,-1):7}
NUM_NEIGHBORS=8
RIGHT_TURN=2
outside = 0
inside = 0.3
even = [0,2,4,6]
mid = 1
height = 40
width = 40
minimum= 1
maximum= 3


def prod(v, i):
    return (v[0]*i, v[1]*i)


def sum_vectors(v1,v2):
    return tuple(map(operator.add, v1,v2))


#True iff (x,y) has valid coordinates
def is_valid(x,y):
    return x>0 and x<width-1 and y>0 and y<height-1


def in_neigh(v, LATTICE):
    count= 0
    for i in range(8):
        n = sum_vectors(v,VECTORS[i])
        if LATTICE[n[0]][n[1]] == inside:
            count+=1
    return count


def space(v, LATTICE, direc, n):
    for i in range(n):
        for j in range(n):
            t = sum_vectors(prod(VECTORS[direc], i), v)
            t1 = sum_vectors(t, prod(VECTORS[(direc+2)%8], j))
            t2 = sum_vectors(t, prod(VECTORS[(direc-2)%8], j))
            if is_valid(t1[0], t1[1]) and LATTICE[t1[0]][t1[1]] == inside:
                return False
            if is_valid(t2[0], t2[1]) and LATTICE[t2[0]][t2[1]] == inside:
                return False
    return True


def fix_corner(LATTICE):
        difference=self.current_vector-self.old_vector
        if abs(difference) in {0,4}:
            return
        reference=self.old_vector
        direction=self.old_vector
        if self.current_vector%2==0:
                reference=self.current_vector
                direction=self.current_vector+4
        for j in range(1,self.thickness-1):
            for i in range(self.thickness):
                for k in {RIGHT_TURN,-RIGHT_TURN}:
                    (x,y)=neighbor(self.current_site[0],self.current_site[1],reference,k,i)
                    self.set_color(neighbor(x,y,direction,0,j)[0],neighbor(x,y,direction,0,j)[1],self.filling,self.is_valid)

        
def create_loop(x,y):
    succ=0
    path=[]
    thickness = randint(3,4)
    LATTICE = [[outside]*height for i in range(width)]
    curr = (x, y)
    for i in range(500):
        direc = 2* randint(0,3)
        r = randint(3,10)
        count = 0
        for l in range(r):
            (a, b) = sum_vectors(curr,VECTORS[direc])
            new = (a, b)
            if is_valid(a, b) and LATTICE[a][b] != inside and in_neigh(new, LATTICE)<=3 and space(new, LATTICE, direc, 10):
                LATTICE[a][b] = inside
                curr = (a, b)
                count+=1
                path.append([curr, direc, count])
        if count>0:
            succ+=1
    for [c, d, e] in path:
        if e!=1:
            for i in range(1,thickness):
                for j in {2,-2}:
                    (w, z) = sum_vectors(c,(i*VECTORS[(d+j)%8][0], i*VECTORS[(d+j)%8][1]))
                    if is_valid(w,z):
                        LATTICE[w][z] = inside
        if e==1:
            for i in range(-thickness+1, thickness):
                for j in range(-thickness+1, thickness):
                    (w, z) = sum_vectors(c,(i,j))
                    if is_valid(w,z):
                        LATTICE[w][z] = inside
    return (LATTICE, succ)


def fix(LATTICE):
    for x in range(width):
        for y in range(height):
            if LATTICE[x][y]== outside:
                for i in range(8):
                    (a, b) = sum_vectors((x,y),VECTORS[i])
                    if is_valid(a,b) and LATTICE[a][b] == inside:
                        LATTICE[x][y] = mid
                        break
    for x in range(width):
        for y in range(height):
            if LATTICE[x][y]== inside:
                LATTICE[x][y]= outside
    return LATTICE
                    

def gnd(LATTICE):
    LATTICE1 = [[outside]*height for i in range(width)]
    for x in range(width):
        for y in range(height):
            if LATTICE[x][y]== inside:
                LATTICE1[x][y] = 0
            else:
                LATTICE1[x][y] = 1
    return LATTICE1                       


''' 
def image():
    var= True
    while var:
        (LATTICE, succ) = create_loop(12,12)
        if succ>20:
            var= False
            LATTICE1=np.array(gnd(LATTICE), dtype=np.uint8)
            LATTICE = fix(LATTICE)
            image = np.array(LATTICE, dtype=np.uint8)
            #plt.imshow(image, interpolation='none')
            plt.imshow(LATTICE1, interpolation='none')
            plt.show()
'''

def create_data_set():
    start_x=randint(30/3,2*30/3)
    start_y=randint(30/3,2*30/3)
    var= True
    while var:
        (LATTICE, succ) = create_loop(start_x, start_y)
        if succ>20:
            var = False
            LATTICE1 = gnd(LATTICE)
            LATTICE = fix(LATTICE)

    LATTICE1 = 1 - np.uint8(LATTICE1)
    LATTICE1 = np.pad(LATTICE1, ((1,1), (1,1)), 'constant', constant_values=((0, 0), (0, 0)))
    LATTICE = np.pad(LATTICE, ((1, 1), (1, 1)), 'constant', constant_values=((0, 0), (0, 0)))

    return LATTICE, LATTICE1

