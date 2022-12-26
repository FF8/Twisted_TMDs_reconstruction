'''
Author: Fabio Ferreira
Created: 16/11/2020
Modified: 04/12/2020
'''

import numpy as np
from numpy import pi, sqrt, sin, cos, exp, cosh
from numpy.linalg import inv
import configparser
import sys

from derivative_adhenergy import derivative_W


config = configparser.ConfigParser()

config.read('config.ini')  #Read config.ini file

material = config['material']

jobs = config['jobs']

#Check what is the material

if (material.get("material") == 'MoS2'):

    print("Calculation for MoS2")
    mat_config = config['MoS2']

elif( material.get("material") == 'WSe2'):

    print("Calculation for WSe2")
    mat_config = config['WSe2']
elif( material.get("material") == 'WS2'):

    print("Calculation for WS2")
    mat_config = config['WS2']

elif( material.get("material") == 'MoSe2'):

    print("Calculation for MoSe2")
    mat_config = config['MoSe2']



#################################

####   Constants here  ####


a = float(mat_config.get('lattice_constant'))    #Lattice constant Angstrom
at = a   # Lattice constant top layer  Angstrom
ab = a   # Lattice constant bot layer  Angstrom


d0 = float(mat_config.get('interlayer_distance'))#   Interlayer distance  Angstrom


q =  float(mat_config.get('q'))/(10)                   #Ang^-1
A1 = float(mat_config.get('A1'))/(10**2)               #eV/ Ang^-2
A2 = float(mat_config.get('A2'))/(10**2)               #eV/Ang^-2
epsilon = float(mat_config.get('epsilon'))/(10**4)     #eV/Ang^-4


oo = float(mat_config.get('poisson_ratio')) #poisson ratio
YM = float(mat_config.get('Young_mod'))     #Young's module




lambda_t = (YM*d0*oo)/ ( (1+oo)*(1-2*oo))  #Lamé top layer coefficient

lambda_t = lambda_t*1e9*6.242*1e+18/(1e10)**3    #GPa-> Pa  J->eV   m->Angstrom

lambda_b = (YM*d0*oo)/ ( (1+oo)*(1-2*oo))   #Lamé bot coefficient

lambda_b = lambda_b*1e9*6.242*1e+18/(1e10)**3



mu_t = YM*d0/(2*(1+oo))                   #shear module  top
mu_t = mu_t*1e9*6.242*1e+18/(1e10)**3     #GPa -> Pa,  J-> eV,   m->Angstrom

mu_b = YM*d0/(2*(1+oo))
mu_b = mu_b*1e9*6.242*1e+18/(1e10)**3




stack_orientation = config['orientation']


if (stack_orientation.get('stack')=='P'):
    phi = float(pi/2)
    print("This is a P calculation")

else:
    phi=0
    print("This is a AP calculation")


angle_parameter = float(stack_orientation.get('angle'))


theta = angle_parameter*pi/(180.0)   #rads


delta = (at/ab-1)   # lattice mismatch

theta_mat = np.matrix([[0,  -theta], [theta, 0]])
W = np.matrix([[delta, theta], [-theta, delta]])
W2 = np.matrix([[delta, -theta], [theta, delta]])

print("W matrix:", W)

l = a/(sqrt(delta**2+theta**2))
print("l (lattice constant of moire supercell (ANG)) is: ", l)

print("\n")



G1_v = np.matrix([ [2*pi/(a)], [2*pi/(a*sqrt(3))] ])

G2_v = np.matrix([ [-2*pi/(a)], [2*pi/(a*sqrt(3))] ])

G3_v = np.matrix([ [0], [-4*pi/(a*sqrt(3))]]   )





if G3_v.all() != -1*(G1_v+G2_v).all():
    print("!!!! G3 not well defined !!!!")


g1_v = delta*G1_v - theta_mat*G1_v          #moire g-vectors
g2_v = delta*G2_v - theta_mat*G2_v
g3_v = delta*G3_v - theta_mat*G3_v


#g1_v = W*G1_v
#g2_v = W*G2_v
#g3_v = W*G3_v



G1 = np.array([ float(G1_v[0]), float(G1_v[1])])        #converts G's to arrays 
G2 = np.array([ float(G2_v[0]), float(G2_v[1])])
G3 = np.array([ float(G3_v[0]), float(G3_v[1])])


if  g3_v.all() != -1*(g1_v+g2_v).all():                             #check if g3  is = -1(g1+g2)   
    #print("!!!!!! Confirm that g3 is well defined !!!!!")
    print(g3_v, -1*(g1_v+g2_v))                     



a1 = np.array([ 1/2 *a, sqrt(3)/2 *a])
a2 = np.array([ -1/2 *a, sqrt(3)/2 *a])


print("a1 vector: ", a1, "Magnitude of a1: ", np.linalg.norm(a1))
print("a2 vector: ", a2, "Magnitude of a2: ", np.linalg.norm(a2))

print("\n")

print("G1 vector: ", G1, "Magnitude of G1: ", np.linalg.norm(G1))
print("G2 vector: ", G2, "Magnitude of G2: ", np.linalg.norm(G2))
print("G3 vector: ", G3, "Magnitude of G3: ", np.linalg.norm(G3))

print("\n")

print("a1.G1 = ",  np.dot(a1,G1))
print("a2.G2 = ",  np.dot(a2,G2))

print("\n")



a1m = np.matrix([[1/2 *a], [sqrt(3)/2 *a]])   #This is a1 but in matrix form
a2m = np.matrix([[-1/2 *a], [sqrt(3)/2 *a]])  #This is a2 but '...'



A1v = inv(W2)*a1m                                   #moire vectors
A2v = inv(W2)*a2m

A1v = np.array([float(A1v[0]), float(A1v[1])])      #moire vectors array form
A2v = np.array([float(A2v[0]), float(A2v[1])])


g1 = np.array([ float(g1_v[0]), float(g1_v[1])])        #moire g vectors array form
g2 = np.array([ float(g2_v[0]), float(g2_v[1])])
g3 = np.array([ float(g3_v[0]), float(g3_v[1])])


print("A1 vector: ", A1v, "Magnitude of A1: ", np.linalg.norm(A1v))
print("A2 vector: ", A2v, "Magnitude of A2: ", np.linalg.norm(A2v))

print("\n")


print("g1 vector: ", g1, "Magnitude of g1: ", np.linalg.norm(g1))
print("g2 vector: ", g2, "Magnitude of g2: ", np.linalg.norm(g2))
print("g3 vector: ", g3, "Magnitude of g3: ", np.linalg.norm(g3))

print("\n")

print("A1.g1 = ",  np.dot(A1v,g1))          #confirm A_i.g_i = 2pi
print("A2.g2 = ",  np.dot(A2v,g2))

print("\n")

G = np.linalg.norm(G1)                      #magnitude of |G1=G1=G3|

print("|A1| / |a1| = ",np.linalg.norm(A1v)/(np.linalg.norm(a1)) )  #ratio between more vector and lattice vector

#print("A1.g1: ", np.dot(A1,g1))

g1x = g1[0]
g1y = g1[1]

g2x = g2[0]
g2y = g2[1]

g3x = g3[0]
g3y = g3[1]

G1x = G1[0]
G1y = G1[1]

G2x = G2[0]
G2y = G2[1]

G3x = G3[0]
G3y = G3[1]



print("\n\n")


print("G1x, G1y = ", G1x,G1y)
print("G2x, G2y = ", G2x,G2y)
print("G3x, G3y = ", G3x,G3y)

print("\n\n")

print("g1x, g1y = ", g1x,g1y)
print("g2x, g2y = ", g2x,g2y)
print("g3x, g3y = ", g3x,g3y)

print("\n\n")


############################# MESH #################################

grid_resolution = config['resolution']


di=float(grid_resolution.get('dx')) #increment i/j and x/y direction
dj=float(grid_resolution.get('dy'))



ysize=l             #size of the mesh  
xsize=l*sqrt(3)/2

print('xsize = ', xsize)
print('ysize = ', ysize)


print("di and dj from file are:", di,dj)


#The following loops make sure that the grid dimension will coincide with the moire dimensions (this means that dj and di could be changed)

for j in np.arange(dj-0.5,dj+0.5,0.0005):

    ysize=l
    ysize = int(round(ysize/j,0))

    if(ysize%2==0):
        ysize=ysize+1

    if abs((ysize-1)*j-l)<0.01:
        dj=j
        break

for i in np.arange(di-0.5, di+0.5, 0.0005):

    xsize=l*sqrt(3)/2
    xsize = int(round(xsize/i,0))

    if(xsize%2==0):
        xsize=xsize+1

    if abs((xsize-1)*i-l*sqrt(3)/2)<0.01:
        di=i
        break

  
print("di and dj changed to:", di,dj)

print("After division, size is: ", xsize,ysize)


final_l =  (( (ysize-1)*dj/2 )**2+( (xsize-1)*di)**2)**(0.5)
print(  (xsize-1)*di, (ysize-1)*dj/2  )


print("Final_l is (Ang): ",final_l)
print("|Final l - Real l| ", abs(final_l-l))      #Real l is the amplitude of the moire vector |A1=A2|, it has to coincide with the grid's amplitude (final l)

if( abs(final_l-l))>0.2:
    print("Grid dimensions do not coincide with moire supercell")
    sys.exit(0)     #Program will exit if amplitudes do not coincide




#Constants from the discretization scheme
#for top layer
c1=(lambda_t+2*mu_t)/(di**2)
c2=(mu_t)/(dj**2)
c3=(lambda_t + mu_t)/(4*di*dj)

c4=(lambda_t+2*mu_t)/(dj**2)
c5=(mu_t)/(di**2)
c6=c3


#for bottom layer
c7=(lambda_b+2*mu_b)/(di**2)
c8=(mu_b)/(dj**2)
c9=(lambda_b+ mu_b)/(4*di*dj)

c10=(lambda_b+2*mu_b)/(dj**2)
c11=(mu_b)/(di**2)
c12=c9


print("Intial constants\n")
print("a [ANG]=", a)
print("at [ANG]=", at)
print("ab [ANG]=", ab)
print("d_0 [ANG]=", d0)
print("q [ANG^1]=", q)
print("A1 [ev/ANG^2]=", A1)
print("A2 [ev/ANG^2]=", A2)
print("epsilon [ev/ANG^4]=", epsilon)
print("phi =", phi)
print("theta (degree) = ", angle_parameter)
print("theta (radians)=", theta)
print("\n")
print("Poisson ratio: ",(mat_config.get('poisson_ratio')))
print("Young's modulus: ", (mat_config.get('Young_mod')))
print("first Lame coefficient [ev/ANG^2] (top)= ", lambda_t)
print("first Lame coefficient [ev/ANG^2] (bot)= ", lambda_b)
print("second Lame coefficient [ev/ANG^2] (top)= ", mu_t)
print("second Lame coefficient [ev/ANG^2  (bot)= ", mu_b)

print("\n\n")

print("Discretization constants:\n")


print("C1, C2, C3  [ev/Ang^2 * (1/Ang^2)]= ", c1,c2,c3)
print("C4, C5, C6  [ev/Ang^2 * (1/Ang^2)]= ", c4,c5,c6)
print("C7, C8, C9  [ev/Ang^2 * (1/Ang^2)]= ", c7,c8,c9)
print("C10, C11, C12  [ev/Ang^2 * (1/Ang^2)]= ", c10,c11,c12)

print("\n\n")


##################################



#reconstruction field arrays/matrices  (xsize X ysize) given by grid dimensions

utx = np.zeros((int(xsize),int(ysize)))
uty = np.zeros((int(xsize),int(ysize)))
ubx = np.zeros((int(xsize),int(ysize)))
uby = np.zeros((int(xsize),int(ysize)))


#np.load('ubx_array.npy', ubx)
#np.load('utx_array.npy', utx)
#np.load('uby_array.npy', uby)
#np.load('uty_array.npy', uty)


string_conct = str(angle_parameter)+stack_orientation.get('stack')+"_"+material.get("material")+'.npy'

ubx_string = 'ubx_array'+string_conct
utx_string = 'utx_array'+string_conct
uby_string = 'uby_array'+string_conct
uty_string = 'uty_array'+string_conct


import datetime

A1_const = A1 #eV/ Ang^-2
A2_const = A2  #eV/Ang^-2
print("A1 constant [eV/ Ang^-2] = ",A1_const)
print("A2 constant [eV/ Ang^-2] = ",A2_const)




try:                                    #In you case you are restating a calculations, then reconstruction field vectors/matrices will be loaded
    ubx = np.load(ubx_string)
    utx = np.load(utx_string)
    uby = np.load(uby_string)
    uty = np.load(uty_string)

except:
    print("Error loading displacement array. Check if this is ok!")    #This is ok if you are not restarting a calculations

utx = np.zeros((int(xsize),int(ysize)))
uty = np.zeros((int(xsize),int(ysize)))
ubx = np.zeros((int(xsize),int(ysize)))
uby = np.zeros((int(xsize),int(ysize)))


#np.load('ubx_array.npy', ubx)
#np.load('utx_array.npy', utx)
#np.load('uby_array.npy', uby)
#np.load('uty_array.npy', uty)


string_conct = str(angle_parameter)+stack_orientation.get('stack')+"_"+material.get("material")+'.npy'

ubx_string = 'ubx_array'+string_conct
utx_string = 'utx_array'+string_conct
uby_string = 'uby_array'+string_conct
uty_string = 'uty_array'+string_conct



text_file = open("Output_vec.txt", "w")


'''
Function lattice_conditions
Correspondence between the rectangular grid indexes (k,l) with the lattice positions (i,j)
This function returns i and j
'''



def lattice_conditions(k,l,xsize,ysize,di,dj):

        shift_h = 0#-(xsize-1)*di*sqrt(3)*(1/3)
        shift_v = 0#(xsize-1)*di

        ys = int((ysize-1)/2)
        xs = xsize-1        

        j = l - ys
        i = k

        i=i*di
        j=j*dj



        j = j + shift_h

        i = i + shift_v

        text_file.write('region1: '+ str(k)+'    '+ str(l)+'    '+ str(i)+'    '+ str(j)+'\n')

        return i,j



import datetime

A1_const = A1 #eV/ Ang^-2                           #Constants in adhesion energy
A2_const = A2  #eV/Ang^-2
print("A1 constant [eV/ Ang^-2] = ",A1_const)
print("A2 constant [eV/ Ang^-2] = ",A2_const)
try:
    ubx = np.load(ubx_string)
    utx = np.load(utx_string)
    uby = np.load(uby_string)
    uty = np.load(uty_string)

except:
    print("Error loading displacement array. Check if this is ok!")

if (jobs.get("compute_reconstruction") == 'yes'):

    print("Started at: " + str(datetime.datetime.now().time()))

    exit_it = False



    #Important: We never start with the true value of A1 and A2.  We increase it step by step.
    #We first find the solution for a small A1 and A2=0 (or vice versa),  and then when we reach A1=A1_true,  we increase A2 up to A2_true.
    #This was suggested by Sergey Slizovskiy, and worked like a charm.

    iter_array=np.array([0.2,3,7,18,38,45,50,67,88,100])


    for km in range(len(iter_array)*2):

        #km=km+0

        try:
            if km<len(iter_array):                  
                value1 = 0*iter_array[km]
                value2 = iter_array[km]
                print("km =", km, "iter[km] = ", iter_array[km], value1,value2)
            else:
                value2 = 100
                value1 = 0.1+iter_array[km-len(iter_array)]
                print("km =", km, "iter[km] = ", iter_array[km-len(iter_array)], value1,value2)


        except:
            print("All iterations have been computed")
            exit_it = True



        count_eq = 0

        A1=value1*A1_const/100              #A1 is increased step by step, see value1 above
        A2=value2*A2_const/100              #A2 is increases step by step, see value2 above

        print("This is the iteration: ", km, "\n")
        print("There are ", len(iter_array)*2 ," iterations\n")


        print("A1,A2: " + str(A1)+ ", " + str(A2)+ ", ")


        success_solve = False

        try:
            ubx = np.load(ubx_string)
            utx = np.load(utx_string)
            uby = np.load(uby_string)
            uty = np.load(uty_string)

            print("Array has been loaded!")

            #print(ubx)

            if (exit_it==True):
                print("Exiting the program")
                break


        except:
            print("Error loading displacement array. Check if this is ok!")


        ####################################################################
        ###################################################################

        #HERE, VARIABLES AND EQUATIONS ARE WRITTEN TO "equations_writer_....py"#
        
        
        #This will be read by gekko  in order to solve the system of equations

        import sympy
        #from sympy import *
        import numpy as np

        
        #Everythng (initial values for variables and equations)  will be written in the equation_writer_....py
        
        eq_writer = "equations_writer_" + str(material.get("material")) + "_" + str(angle_parameter)+stack_orientation.get('stack') + ".py"

        f = open(eq_writer, "w")


        print("SIZE: ", xsize, ysize)


        f.write("#from sympy import *\n\n\n")
        f.write("import random\n")

        c=0

        # Following four loops -> write initial constraints  for displacemente field vectors

        for i in range(xsize):
            for j in range(ysize):
        #        temp_string = 'utx'+str("{:02d}".format(i))+'_'+str("{:02d}".format(j))+'= m.Var(value=0.1)'
        #        temp_string = 'utx'+str(i)+'_'+str(j)+'= m.Var(value=random.uniform(-1,1))'
        #        temp_string = 'utx'+str(i)+'_'+str(j)+'= m.Var(value=0.1)'
                temp_string1 = 'utx'+str(i)+'_'+str(j)+'= m.Var(value=utx['+str(i)+','+str(j)+'])'
                temp_string2 = 'utx'+str(i)+'_'+str(j)+'.upper = 1'
                temp_string3 = 'utx'+str(i)+'_'+str(j)+'.lower = -1'
                c=c+1
                #print(temp_string)
                f.write(temp_string1+"\n")
                f.write(temp_string2+"\n")
                f.write(temp_string3+"\n")


        for i in range(xsize):
            for j in range(ysize):
        #        temp_string = 'uty'+str("{:02d}".format(i))+'_'+str("{:02d}".format(j))+'= m.Var(value=0.1)'
        #        temp_string = 'uty'+str(i)+'_'+str(j)+'= m.Var(value=random.uniform(-1,1))'
        #        temp_string = 'uty'+str(i)+'_'+str(j)+'= m.Var(value=0.1)'
                temp_string1 = 'uty'+str(i)+'_'+str(j)+'= m.Var(value=uty['+str(i)+','+str(j)+'])'
                temp_string2 = 'uty'+str(i)+'_'+str(j)+'.upper = 1'
                temp_string3 = 'uty'+str(i)+'_'+str(j)+'.lower = -1'
                c=c+1
                #print(temp_string)
                f.write(temp_string1+"\n")
                f.write(temp_string2+"\n")
                f.write(temp_string3+"\n")


        for i in range(xsize):
            for j in range(ysize):
        #        temp_string = 'ubx'+str("{:02d}".format(i))+'_'+str("{:02d}".format(j))+'= m.Var(value=0.1)'
        #        temp_string = 'ubx'+str(i)+'_'+str(j)+'= m.Var(random.uniform(-1,1))'
        #        temp_string = 'ubx'+str(i)+'_'+str(j)+'= m.Var(value=0.1)'
                temp_string1 = 'ubx'+str(i)+'_'+str(j)+'= m.Var(value=ubx['+str(i)+','+str(j)+'])'
                temp_string2 = 'ubx'+str(i)+'_'+str(j)+'.upper = 1'
                temp_string3 = 'ubx'+str(i)+'_'+str(j)+'.lower = -1'
                c=c+1
                #print(temp_string)
                f.write(temp_string1+"\n")
                f.write(temp_string2+"\n")
                f.write(temp_string3+"\n")


        for i in range(xsize):
            for j in range(ysize):
        #        temp_string = 'uby'+str("{:02d}".format(i))+'_'+str("{:02d}".format(j))+'= m.Var(value=0.1)'
        #        temp_string = 'uby'+str(i)+'_'+str(j)+'= m.Var(random.uniform(-1,1))'
        #        temp_string = 'uby'+str(i)+'_'+str(j)+'= m.Var(value=0.1)'
                temp_string1 = 'uby'+str(i)+'_'+str(j)+'= m.Var(value=uby['+str(i)+','+str(j)+'])'
                temp_string2 = 'uby'+str(i)+'_'+str(j)+'.upper = 1'
                temp_string3 = 'uby'+str(i)+'_'+str(j)+'.lower = -1'
                c=c+1
                #print(temp_string)
                f.write(temp_string1+"\n")
                f.write(temp_string2+"\n")
                f.write(temp_string3+"\n")


        print("There are " + str(c) + " variables\n")

        f.write("\n\n\n")



        # following 2 loops write initial conditions and periodic boundary conditions (like stated in my first year report)

        for i in range (1,xsize-1):

            temp_string1='m.Equation(utx'+str(i)+'_'+str(0)+'-utx'+str(i)+'_'+str(ysize-1)+'==0)'
            temp_string2='m.Equation(uty'+str(i)+'_'+str(0)+'-uty'+str(i)+'_'+str(ysize-1)+'==0)'
            temp_string3='m.Equation(ubx'+str(i)+'_'+str(0)+'-ubx'+str(i)+'_'+str(ysize-1)+'==0)'
            temp_string4='m.Equation(uby'+str(i)+'_'+str(0)+'-uby'+str(i)+'_'+str(ysize-1)+'==0)'

            f.write(temp_string1+"\n")
            f.write(temp_string2+"\n")
            f.write(temp_string3+"\n")
            f.write(temp_string4+"\n\n\n")



        for j in range(ysize):
            Nb=int(((ysize-1)/2))


            if (j==0):

                temp_string1='m.Equation(utx'+str(0)+'_'+str(j)+'-utx'+str(0)+'_'+str(ysize-1)+'==0)'
                temp_string2='m.Equation(utx'+str(0)+'_'+str(j)+'-utx'+str(xsize-1)+'_'+str(Nb)+'==0)'

                count_eq = count_eq +2

                temp_string3='m.Equation(uty'+str(0)+'_'+str(j)+'-uty'+str(0)+'_'+str(ysize-1)+'==0)'
                temp_string4='m.Equation(uty'+str(0)+'_'+str(j)+'-uty'+str(xsize-1)+'_'+str(Nb)+'==0)'

                count_eq = count_eq +2

                temp_string5='m.Equation(ubx'+str(0)+'_'+str(j)+'-ubx'+str(0)+'_'+str(ysize-1)+'==0)'
                temp_string6='m.Equation(ubx'+str(0)+'_'+str(j)+'-ubx'+str(xsize-1)+'_'+str(Nb)+'==0)'

                count_eq = count_eq +2

                temp_string7='m.Equation(uby'+str(0)+'_'+str(j)+'-uby'+str(0)+'_'+str(ysize-1)+'==0)'
                temp_string8='m.Equation(uby'+str(0)+'_'+str(j)+'-uby'+str(xsize-1)+'_'+str(Nb)+'==0)'

                count_eq = count_eq +2


                f.write(temp_string1+"\n")
                f.write(temp_string2+"\n")
                f.write(temp_string3+"\n")
                f.write(temp_string4+"\n")

                f.write(temp_string5+"\n")
                f.write(temp_string6+"\n")
                f.write(temp_string7+"\n")
                f.write(temp_string8+"\n\n\n")





            if (j>0 and j<Nb):

                temp_string1='m.Equation(utx'+str(0)+'_'+str(j)+'-utx'+str(xsize-1)+'_'+str(Nb+j)+'==0)'
                temp_string2='m.Equation(uty'+str(0)+'_'+str(j)+'-uty'+str(xsize-1)+'_'+str(Nb+j)+'==0)'
                temp_string3='m.Equation(ubx'+str(0)+'_'+str(j)+'-ubx'+str(xsize-1)+'_'+str(Nb+j)+'==0)'
                temp_string4='m.Equation(uby'+str(0)+'_'+str(j)+'-uby'+str(xsize-1)+'_'+str(Nb+j)+'==0)'

                count_eq = count_eq +4

                #temp_string1='utx'+str(i)+'_'+str(0)+'=utx'+str(Nb+i)+'_'+str(ysize-1)
                #temp_string2='uty'+str(i)+'_'+str(0)+'=uty'+str(Nb+i)+'_'+str(ysize-1)
                #temp_string3='ubx'+str(i)+'_'+str(0)+'=ubx'+str(Nb+i)+'_'+str(ysize-1)
                #temp_string4='uby'+str(i)+'_'+str(0)+'=uby'+str(Nb+i)+'_'+str(ysize-1)



                f.write(temp_string1+"\n")
                f.write(temp_string2+"\n")
                f.write(temp_string3+"\n")
                f.write(temp_string4+"\n\n\n")




            if (j>Nb and j<ysize-1):

                temp_string1='m.Equation(utx'+str(0)+'_'+str(j)+'-utx'+str(xsize-1)+'_'+str(-Nb+j)+'==0)'
                temp_string2='m.Equation(uty'+str(0)+'_'+str(j)+'-uty'+str(xsize-1)+'_'+str(-Nb+j)+'==0)'
                temp_string3='m.Equation(ubx'+str(0)+'_'+str(j)+'-ubx'+str(xsize-1)+'_'+str(-Nb+j)+'==0)'
                temp_string4='m.Equation(uby'+str(0)+'_'+str(j)+'-uby'+str(xsize-1)+'_'+str(-Nb+j)+'==0)'
                count_eq = count_eq +4


                #temp_string1='utx'+str(i)+'_'+str(0)+'=utx'+str(i)+'_'+str(ysize-1)
                #temp_string2='uty'+str(i)+'_'+str(0)+'=uty'+str(i)+'_'+str(ysize-1)
                #temp_string3='ubx'+str(i)+'_'+str(0)+'=ubx'+str(i)+'_'+str(ysize-1)
                #temp_string4='uby'+str(i)+'_'+str(0)+'=uby'+str(i)+'_'+str(ysize-1)


                f.write(temp_string1+"\n")
                f.write(temp_string2+"\n")
                f.write(temp_string3+"\n")
                f.write(temp_string4+"\n\n\n")





            if (j==Nb):

                temp_string1='m.Equation(utx'+str(0)+'_'+str(j)+'-utx'+str(xsize-1)+'_'+str(ysize-1)+'==0)'
                temp_string2='m.Equation(utx'+str(0)+'_'+str(j)+'-utx'+str(xsize-1)+'_'+str(0)+'==0)'

                temp_string3='m.Equation(uty'+str(0)+'_'+str(j)+'-uty'+str(xsize-1)+'_'+str(ysize-1)+'==0)'
                temp_string4='m.Equation(uty'+str(0)+'_'+str(j)+'-uty'+str(xsize-1)+'_'+str(0)+'==0)'

                temp_string5='m.Equation(ubx'+str(0)+'_'+str(j)+'-ubx'+str(xsize-1)+'_'+str(ysize-1)+'==0)'
                temp_string6='m.Equation(ubx'+str(0)+'_'+str(j)+'-ubx'+str(xsize-1)+'_'+str(0)+'==0)'

                temp_string7='m.Equation(uby'+str(0)+'_'+str(j)+'-uby'+str(xsize-1)+'_'+str(ysize-1)+'==0)'
                temp_string8='m.Equation(uby'+str(0)+'_'+str(j)+'-uby'+str(xsize-1)+'_'+str(0)+'==0)'


                count_eq = count_eq +8



                #temp_string1='utx'+str(i)+'_'+str(0)+'=utx'+str(Nb)+'_'+str(ysize-1)
                #temp_string2='uty'+str(i)+'_'+str(0)+'=uty'+str(Nb)+'_'+str(ysize-1)
                #temp_string3='ubx'+str(i)+'_'+str(0)+'=ubx'+str(Nb)+'_'+str(ysize-1)
                #temp_string4='uby'+str(i)+'_'+str(0)+'=uby'+str(Nb)+'_'+str(ysize-1)


                f.write(temp_string1+"\n")
                f.write(temp_string2+"\n")
                f.write(temp_string3+"\n")
                f.write(temp_string4+"\n")

                f.write(temp_string5+"\n")
                f.write(temp_string6+"\n")
                f.write(temp_string7+"\n")
                f.write(temp_string8+"\n\n\n")




    


        #This is a function that defined the rhs of the Euler-Lagrange equations (see 1st year report file). Basically a string containing the equations will be returned.
        #These strings will be written in a file called  equations_writer_yyy.py    yyy ->depend on the material and angle


        def rhs_equation(layer,var,k,l,xsize,ysize,count_eq):


            bool_s = True

            rhs = 'test'

            if layer=='top':


                if var =='x':


                    u_1='utx'
                    u_2='uty'

                    c_1 = 'c1'
                    c_2 = 'c2'
                    c_3 = 'c3'

                if var == 'y':

                    u_1='uty'
                    u_2='utx'

                    c_1 = 'c5'
                    c_2 = 'c4'
                    c_3 = 'c6'


            if layer=='bot':


                if var =='x':

                    u_1='ubx'
                    u_2='uby'

                    c_1 = 'c7'
                    c_2 = 'c8'
                    c_3 = 'c9'

                if var == 'y':

                    u_1='uby'
                    u_2='ubx'

                    c_1 = 'c11'
                    c_2 = 'c10'
                    c_3 = 'c12'



            if (k>0 and k<xsize-1 and l>0 and l<ysize-1):

                rhs='(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+str(k)+'_'+str(l)+\
                '+'+str(c_1)+'*('+str(u_1)+str(k+1)+'_'+str(l)+'+'+str(u_1)+str(k-1)+'_'+str(l)+\
                ')+'+str(c_2)+'*('+str(u_1)+str(k)+'_'+str(l+1)+'+'+str(u_1)+str(k)+'_'+str(l-1)\
                +')+'+str(c_3)+'*('+str(u_2)+str(k+1)+'_'+str(l+1)+'-'+str(u_2)+str(k+1)+'_'+str(l-1)+'+'+str(u_2)+str(k-1)\
                +'_'+str(l-1)+'-'+str(u_2)+str(k-1)+'_'+str(l+1)+')'

                count_eq = count_eq +1

                return rhs,bool_s


            if (k==0 and l==0):


                Nb = int((ysize-1)/2)

                rhs ='(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+str(k)+'_'+str(l)+\
                '+'+str(c_1)+'*('+str(u_1)+str(k+1)+'_'+str(l)+'+'+str(u_1)+str(xsize-2)+'_'+str(Nb)+\
                ')+'+str(c_2)+'*('+str(u_1)+str(k)+'_'+str(l+1)+'+'+str(u_1)+str(k)+'_'+str(ysize-2)+\
                ')+'+str(c_3)+'*('+str(u_2)+str(k+1)+'_'+str(l+1)+'-'+str(u_2)+str(xsize-2)+'_'+str(Nb+1)+'+'+str(u_2)+str(xsize-2)\
                +'_'+str(Nb-1)+'-'+str(u_2)+str(k+1)+'_'+str(ysize-2)+')'

                count_eq = count_eq +1


                return rhs,bool_s


            if (k==0 and l==ysize-1):

                #This condition is equal to (k==0 and l==0).
                #It can be ignored provided that periodic conditions are well defined.


                rhs='nothing'
                bool_s=False

                #rhs = '(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+  str(k)+ '_'+str(l) + ' + '+str(c_1)+'*('+str(u_1) + str(1)+ '_'+str(l) + '+'+str(u_1) + str(k-1)+ '_'+str(l) + ') + '+str(c_2)+'*('+str(u_1) +\
                #    str(k)+ '_'+str(l+1) + '+'+str(u_1) + str((int((xsize-1)/2)))+ '_'+str(ysize-2) + ') + '+str(c_3)+'*('+str(u_2) + str(1)+ '_'+str(l+1) + ' -'+str(u_2) + str((int((xsize-1)/2)+1))+ '_'+str(ysize-2) + '+'+str(u_2) +\
                #    str((int((xsize-1)/2)-1))+ '_'+str(ysize-2) +   '- '+str(u_2) + str(k-1)+ '_'+str(l+1) +')'

                return rhs,bool_s




            if (l==0 and k>0 and k<xsize-1):


                rhs='(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+str(k)+'_'+str(l)+'+'+str(c_1)+\
                '*('+str(u_1)+str(k+1)+'_'+str(l)+'+'+str(u_1)+str(k-1)+'_'+str(l)+')+'+str(c_2)+\
                '*('+str(u_1)+str(k)+'_'+str(l+1)+'+'+str(u_1)+str(k)+'_'+str(ysize-2)+')+'+str(c_3)+\
                '*('+str(u_2)+str(k+1)+'_'+str(l+1)+'-'+str(u_2)+str(k+1)+'_'+str(ysize-2)+'-'+str(u_2)+\
                str(k-1)+'_'+str(l+1)+'+'+str(u_2)+str(k-1)+'_'+str(ysize-2)+')'

                #rhs = '('+str(u_1)+str(k)+'_'+str(l)+')'

                count_eq = count_eq +1

                return rhs,bool_s



            if (l==ysize-1 and k>0 and k<xsize-1):


                #This condition is equal to (l==0 and k=>0 and k<ysize-1). x\
                #It can be ignored provided that periodic conditions are well defined.


                bool_s = False

                #rhs = '(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+  str(k)+ '_'+str(l) + ' + '+str(c_1)+'*('+str(u_1) + str(1)+ '_'+str(l) + '+'+str(u_1) + str(k-1)+ '_'+str(l) + ') + '+str(c_2)+'*('+str(u_1) +\
                #    str(k)+ '_'+str(l+1) + '+'+str(u_1) + str(k)+ '_'+str(l-1) + ') + '+str(c_3)+'*('+str(u_2) + str(1)+ '_'+str(l+1) + ' -'+str(u_2) + str(1)+ '_'+str(l-1) + '+'+str(u_2) +\
                #    str(k-1)+ '_'+str(l-1) +   '- '+str(u_2) + str(k-1)+ '_'+str(l+1) +')'

                rhs = 'nothing'

                return rhs,bool_s



            if (l==0 and k==xsize-1):


                #THIS IS THE FIXED POINT!!

                bool_s =False  #Change to false if you want to fixe the point

                #rhs='nothing'
                Nb = int((ysize-1)/2)


                #rhs ='(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+str(k)+'_'+str(l)+'+'+str(c_1)+'*('+str(u_1)+str(k+1)+'_'+str(l)+'+'+str(u_1)+str(xsize-2)+'_'+str(l)+')+'+str(c_2)+'*('+str(u_1)+str(Nb)+'_'+str(ysize-(ysize-1))+'+'+str(u_1)+str(k)+'_'+str(l-1)+')+'+str(c_3)+'*('+str(u_2)+str(Nb+1)+'_'+str(ysize-(ysize-1))+'-'+str(u_2)+str(k+1)+'_'+str(l-1)+'+'+str(u_2)+str(xsize-2)+'_'+str(l-1)+'-'+str(u_2)+str(Nb-1)+'_'+str(ysize-(ysize-1))+')'

                rhs = '('+str(u_1)+str(k)+'_'+str(l)+')'

                count_eq = count_eq +1

                return rhs,bool_s




            if (k==xsize-1 and l==ysize-1):

                bool_s = False


                Nb = int((ysize-1)/2)

                rhs='nothing'
                #continue

                #rhs = '(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+ str(k)+ '_'+str(l) + ' + '+str(c_1)+'*('+str(u_1) + str(1)+ '_'+str(l) + '+'+str(u_1) + str(k-1)+ '_'+str(l) + ') + '+str(c_2)+'*('+str(u_1) +\
                #    str((int((xsize-1)/2)))+ '_'+str(1) + '+'+str(u_1) + str(k)+ '_'+str(l-1) + ') + '+str(c_3)+'*('+str(u_2) + str((int((xsize-1)/2)+1))+ '_'+str(1) + ' -'+str(u_2) + str(1)+ '_'+str(l-1) + '+'+str(u_2) +\
                #    str(k-1)+ '_'+str(l-1) +   '- '+str(u_2) + str((int((xsize-1)/2))-1)+ '_'+str(1) +')'


                #rhs = '('+str(u_1)+str(k)+'_'+str(l)+')'

                return rhs,bool_s




            if (l>0 and l<ysize-1 and k==0):


                #bool_s = False

                Nb = int((ysize-1)/2)

                if (l>0 and l<Nb):

                    rhs = '(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+  str(k)+ '_'+str(l) + ' + '+\
                    str(c_1)+'*('+str(u_1) + str(k+1)+ '_'+str(l) + '+'+str(u_1) + str(xsize-2)+ '_'+str(Nb+l) + ') + '+\
                    str(c_2)+'*('+str(u_1) +str(k)+ '_'+str(l+1) + '+'+str(u_1) + str(k)+ '_'+str(l-1) + ') + '+\
                    str(c_3)+'*('+str(u_2) + str(k+1)+ '_'+str(l+1) + ' -'+str(u_2) + str(xsize-2)+ '_'+str(Nb+l+1) +\
                    '+'+str(u_2) +str(xsize-2)+ '_'+str(Nb+l-1) +   '- '+str(u_2) + str(k+1)+ '_'+str(l-1) +')'

                    count_eq = count_eq +1

                    return rhs,bool_s

                if (l>Nb and l<ysize-1):


                    rhs = '(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+ str(k)+ '_'+str(l) + ' + '+\
                    str(c_1)+'*('+str(u_1) + str(k+1)+ '_'+str(l) + '+'+str(u_1) + str(xsize-2)+ '_'+str(l-Nb) + ') + '+\
                    str(c_2)+'*('+str(u_1) +str(k)+ '_'+str(l+1) + '+'+str(u_1) + str(k)+ '_'+str(l-1) + ') + '+\
                    str(c_3)+'*('+str(u_2) + str(k+1)+ '_'+str(l+1) + ' -'+str(u_2) + str(xsize-2)+ '_'+str(l+1-Nb) + '+'+\
                    str(u_2) +str(xsize-2)+ '_'+str(l-1-Nb) +   '- '+str(u_2) + str(k+1)+ '_'+str(l-1) +')'

                    count_eq = count_eq +1

                    return rhs,bool_s



                if (l==Nb):

                    bool_s=False

                    rhs='nothing'

                    #rhs = '('+str(u_1)+str(k)+'_'+str(l)+')'

                    #rhs = '(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+  str(k)+ '_'+str(l) + ' + '+str(c_1)+'*('+str(u_1) + str(k+1)+ '_'+str(l) + '+'+str(u_1) + str(k-1)+ '_'+str(l) + ') + '+str(c_2)+'*('+str(u_1) +\
                    #str(k)+ '_'+str(l+1) + '+'+str(u_1) + str(0)+ '_'+str(ysize-2) + ') + '+str(c_3)+'*('+str(u_2) + str(k+1)+ '_'+str(l+1) + ' -'+str(u_2) + str(1)+ '_'+str(ysize-2) + '+'+str(u_2) +\
                    #str(xsize-2)+ '_'+str(ysize-2) +   '- '+str(u_2) + str(k-1)+ '_'+str(l+1) +')'



                    return rhs,bool_s



            if (l>0 and l<ysize-1 and k==xsize-1):



                Nb = int((ysize-1)/2)

                if (l>0 and l<Nb):    #This condition is equal to (l>Nb and l<xsize and k==0).
                                      #It can be ignored provided that periodic conditions are well defined.


                    bool_s = False
                    rhs='nothing'


                    #rhs = '(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+  str(k)+ '_'+str(l) + ' + '+str(c_1)+'*('+str(u_1) + str(k+1)+ '_'+str(l) + '+'+str(u_1) + str(k-1)+ '_'+str(l) + ') + '+str(c_2)+'*('+str(u_1) +\
                    #str(Nb +k)+ '_'+str(1) + '+'+str(u_1) + str(k)+ '_'+str(l-1) + ') + '+str(c_3)+'*('+str(u_2) + str(Nb+k+1)+ '_'+str(1) + ' -'+str(u_2) + str(k+1)+ '_'+str(l-1) + '+'+str(u_2) +\
                    #str(k-1)+ '_'+str(l-1) +   '- '+str(u_2) + str(Nb+k-1)+ '_'+str(1) +')'



                    return rhs,bool_s


                if (l==Nb):

            #        #continue

                    rhs='nothing'

                    #rhs = '(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1)+ str(k)+ '_'+str(l) + ' + '+str(c_1)+'*('+str(u_1) + str(k+1)+ '_'+str(l) + '+'+str(u_1) + str(k-1)+ '_'+str(l) + ') + '+str(c_2)+'*('+str(u_1) +\
                    #str(0)+ '_'+str(1) + '+'+str(u_1) + str(k)+ '_'+str(l-1) + ') + '+str(c_3)+'*('+str(u_2) + str(1)+ '_'+str(1) + ' -'+str(u_2) + str(k+1)+ '_'+str(l-1) + '+'+str(u_2) +\
                    #str(k-1)+ '_'+str(l-1) +   '- '+str(u_2) + str(xsize-2)+ '_'+str(1) +')'

                    bool_s=False

                    return rhs,bool_s


                if (l>Nb and l<ysize-1):     #This condition is equal to (l>0 and l<Nb and k==0).
                                             #It can be ignored provided that periodic conditions are well defined.

                    rhs ='nothing'
            #        #continue

                    #rhs = '(-2*'+str(c_1)+'-2*'+str(c_2)+')*'+str(u_1) + str(k)+ '_'+str(l) + ' + '+str(c_1)+'*('+str(u_1) + str(k+1)+ '_'+str(l) + '+'+str(u_1) + str(k-1)+ '_'+str(l) + ') + '+str(c_2)+'*('+str(u_1) +\
                    #str(k-Nb)+ '_'+str(1) + '+'+str(u_1) + str(k)+ '_'+str(l-1) + ') + '+str(c_3)+'*('+str(u_2) + str(k+1-Nb)+ '_'+str(1) + ' -'+str(u_2) + str(k+1)+ '_'+str(l-1) + '+'+str(u_2) +\
                    #str(k-1)+ '_'+str(l-1) +   '- '+str(u_2) + str(k-1-Nb)+ '_'+str(1) +')'

                    bool_s=False

                    return rhs,bool_s







        #following loop will write the final  equations 



        for k in range(xsize):
            for l in range(ysize):


                i,j=lattice_conditions(k,l,xsize,ysize,di,dj)


                lhs,bool_s = rhs_equation('top','x',k,l,xsize,ysize,count_eq)




                if (bool_s==True):
                    #print(count)

                    temp_string = derivative_W('top','x',lhs,k,l,i,j)

                    f.write(temp_string+"\n\n\n")

                else:

                    if(lhs!='nothing'):

                        temp_string = 'm.Equation('+str(lhs)+'==0.000)'
                        #temp_string = 'm.Equation('+str(lhs)+'==0.00047448619187965494)'
                        f.write(temp_string+"\n\n\n")

                    else:
                        pass



        for k in range(xsize):
            for l in range(ysize):


                i,j=lattice_conditions(k,l,xsize,ysize,di,dj)

                lhs,bool_s = rhs_equation('top','y',k,l,xsize,ysize,count_eq)

                if (bool_s==True):


                    temp_string = derivative_W('top','y',lhs,k,l,i,j)

                    f.write(temp_string+"\n\n\n")

                else:

                    if(lhs!='nothing'):

                        temp_string = 'm.Equation('+str(lhs)+'==0.000)'
                        f.write(temp_string+"\n\n\n")

                    else:
                        pass


        for k in range(xsize):
            for l in range(ysize):


                i,j=lattice_conditions(k,l,xsize,ysize,di,dj)
                lhs,bool_s = rhs_equation('bot','x',k,l,xsize,ysize,count_eq)

                if (bool_s==True):

                    temp_string = derivative_W('bot','x',lhs,k,l,i,j)
                    f.write(temp_string+"\n\n\n")

                else:

                    if(lhs!='nothing'):

                        temp_string = 'm.Equation('+str(lhs)+'==0.0000)'
                        f.write(temp_string+"\n\n\n")

                    else:
                        pass



        for k in range(xsize):
            for l in range(ysize):


                i,j=lattice_conditions(k,l,xsize,ysize,di,dj)
                lhs,bool_s = rhs_equation('bot','y',k,l,xsize,ysize,count_eq)

                if (bool_s==True):

                    temp_string = derivative_W('bot','y',lhs,k,l,i,j)

                    f.write(temp_string+"\n\n\n")


                else:

                    if(lhs!='nothing'):

                        temp_string = 'm.Equation('+str(lhs)+'==0.0000)'
                        #temp_string = 'm.Equation('+str(lhs)+'==-0.09685803604429122)'
                        f.write(temp_string+"\n\n\n")

                    else:
                        pass



        f.close()
        print("EQUATIONS HAVE BEEN WRITTEN")
        print("(Expected) Number of eq: ", xsize*ysize*4)


        #print(utx[35,31])


        import time

        print("Calling solver: " + str(datetime.datetime.now().time()))

        #SOLVE THE NON LINEAR SYSTEM HERE#
        from gekko import GEKKO

        start_t = time.time()

        m=GEKKO(remote=False)

        m.options.SOLVER=2

        exec(open(eq_writer).read())

        end_eq_t = time.time()

        print("Equations read in (s):", end_eq_t-start_t )

        print('--------- Follow local path to view files --------------')
        print(m.path)               # show source file path
        print('--------------------------------------------------------')


        #m.options.MAX_MEMORY = 6
        #m.options.IMODE = 2

        m.solve()

        end_sol_t = time.time()


        print("Solver has finished in ",end_sol_t-start_t )


        success_solve =True


        #xsize=int(xsize/di)

        #ysize=int(ysize/dj)

        m=int(0)

        #print(xsize)
        #print(ysize)


        f = open("vec_out.py", "w")

        f.write("#from sympy import *\n\n\n")
        f.write("import numpy as np \n")


        f.write('utx = np.zeros((int(xsize),int(ysize)))\n')
        f.write('uty = np.zeros((int(xsize),int(ysize)))\n')
        f.write('ubx = np.zeros((int(xsize),int(ysize)))\n')
        f.write('uby = np.zeros((int(xsize),int(ysize)))\n')



        for i in range(xsize):
            for j in range(ysize):
                temp_string = 'utx['+str(i)+','+str(j)+']=utx'+str(i)+'_'+str(j)+'[0]'

                f.write(temp_string+"\n")

        for i in range(xsize):
            for j in range(ysize):
                temp_string = 'uty['+str(i)+','+str(j)+']=uty'+str(i)+'_'+str(j)+'[0]'
                f.write(temp_string+"\n")

        for i in range(xsize):
            for j in range(ysize):
                temp_string = 'ubx['+str(i)+','+str(j)+']=ubx'+str(i)+'_'+str(j)+'[0]'
                f.write(temp_string+"\n")

        for i in range(xsize):
            for j in range(ysize):
                temp_string = 'uby['+str(i)+','+str(j)+']=uby'+str(i)+'_'+str(j)+'[0]'
                f.write(temp_string+"\n")


        f.close()

        print("THE vec output has been written")
        exec(open('vec_out.py').read())


        if (success_solve == True):


            np.save(ubx_string, ubx)
            np.save(utx_string, utx)
            np.save(uby_string, uby)
            np.save(uty_string, uty)


            print("ARRAY HAS BEEN SAVED.")


        success_solve = False

        print("Going go next iteration: " + str(datetime.datetime.now().time()))

        print("ITERATION NUMBER " + str(km) +" done\n" )



####PLOT HERE"


###This is going to plot in your personal computer###


if (jobs.get("save_u_field") == 'yes'):

    ubx=np.load(ubx_string)
    utx=np.load(utx_string)
    uby=np.load(uby_string)
    uty=np.load(uty_string)


    string_conct2 = str(angle_parameter)+stack_orientation.get('stack')+"_"+material.get("material")+'.dat'


    file_top_name = 'vecfield_sol_top' + string_conct2
    file_bot_name = 'vecfield_sol_bot' + string_conct2

    text_file_vec_top = open(file_top_name, "w")


    print("ys and xs: " + str(dj*(ysize-1)/2) + ", " + str(di*(xsize-1)))



    with open(file_top_name, "w") as fp:

        for k in range(xsize):
            for l in range(ysize):

                i,j=lattice_conditions(k,l,xsize,ysize,di,dj)

                fp.write(str(i)+' '+ str(j) +' '+  str(utx[k,l]) +' '+ str(uty[k,l]) +'\n')



    with open(file_bot_name, "w") as fp:


        for k in range(xsize):
            for l in range(ysize):

                i,j=lattice_conditions(k,l,xsize,ysize,di,dj)

                fp.write(str(i)+' '+ str(j) +' '+  str(ubx[k,l]) +' '+ str(uby[k,l]) +'\n')




    xrt, yrt, fxt, fyt = np.genfromtxt(file_top_name, unpack=True)
    xrb, yrb, fxb, fyb = np.genfromtxt(file_bot_name, unpack=True)


    print(min(xrt), max(xrt))
    print(min(yrt), max (yrt))

    #plt.xlabel("Angstrom")

    
    try:

        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(2,figsize=(10, 20))


        ax[0].tick_params(axis='both', which='major', labelsize=40)
        ax[1].tick_params(axis='both', which='major', labelsize=40)

        ax[0].set_xlabel('ux [Ang]', fontsize=40)
        ax[0].set_ylabel('uy [Ang]', fontsize=40)


        ax[1].set_xlabel('ux [Ang]', fontsize=40)
        ax[1].set_ylabel('uy [Ang]', fontsize=40)




        ax[0].quiver(xrt,yrt,fxt,fyt)
        ax[1].quiver(xrb,yrb,fxb,fyb)


        print('vecfield' + str(angle_parameter)+stack_orientation.get('stack')+"_"+material.get("material")+'.png')
        fig.savefig('vecfield' + str(angle_parameter)+stack_orientation.get('stack')+"_"+material.get("material")+'.png',bbox_inches='tight',dpi=60)

    except:
        
        print("There was an error saving the .png figure. This might have happened due to fact that this code was executed in a cluster or something?")

    

    #Replicate the obtained solution
    
    #try:

    # replicate_lattice(file_top_name,file_bot_name,dj,di,ysize,xsize,final_l, string_conct2)
    
    #except:
    #    print("Calling replicate_lattice didn't work")


import os

try:
    os.remove("equations_writer.py")
except:
    print("Something went wrong when deleting equations_writer.py file (Ignore this message)")        #Ignore this. 
