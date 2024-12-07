#Common Functions used in the Plasma Python code
# All functions being used for plasma_1D and plasma_2d
import math
import cmath
import numpy as np
from scipy.interpolate import interp1d

#dataHandling
# this function reads in the file and takes out the data from the input file to 
# a list to be siphoned through for the data points for later processing
# INPUTS: 
#   filename: a string of the name of the file with all the data. NO FILETYPE ONLY NAME
#   data: an empty list the data from the input file will be appended in
# OUTPUTS: 
#   colNumList: A list with each index holding an integer of the column of where that data is
#   If there is no title of that data found, like no z value meaning it is 2D data
#   the a -1 will be in that spot representing no index
#   index 0 is Plasma Frequency
#   index 1 is Collision Frequency
#   index 2 is x
#   index 3 is y
#   index 4 is z
#   data: All data from the input file in this 2D list
def dataHandling(filename): #inFile
    dataFile = open(filename+".csv" , "r")
    lines = []
    data = []
    colNumList = [-1]*5
    column= 0
    prevComma = 0
    i=0
    j=0
    while True:
        theline = dataFile.readline()
        if theline == "":
            break
        lines.append(theline)
    #data = [[]]*len(lines)
    while j < len(lines[i]):
        if (lines[i][j] == ',') or (j+1 == len(lines[i])):
            last = j
            if lines[i][prevComma:last] == 'PlasmaFreq':
                colNumList[0] = column
            elif lines[i][prevComma:last] == 'CollFreq':
                colNumList[1] = column
            elif lines[i][prevComma:last] == 'Distance':
                colNumList[2] = column
            elif lines[i][prevComma:last] == 'x':
                colNumList[2] = column
            elif lines[i][prevComma:last] == 'y':
                colNumList[3] = column
            elif lines[i][prevComma:last] == 'z':
                colNumList[4] = column
            prevComma = j+1
            column+=1
        j+=1
    prevComma = 0
    i=1
    j=0
    while i < len(lines):
        linesdata = []
        while j < len(lines[i]):
            if (lines[i][j] == ',') or (j+1 == len(lines[i])):
                last = j
                linesdata.append(lines[i][prevComma:last])
                prevComma = j+1
            j+=1
        data.append(linesdata)
        i+=1
        j=0
        prevComma = 0
    dataFile.close()
    return colNumList,data

#dataHandling2D
# this function takes the lines of code and puts it into meshgrids for the plasma 
# and collision frequency and a np.linspace for the x and y values
# INPUTS: 
#   datalist: a 2D list 
#   listofColumns: A list with each index holding an integer of the column of where 
#   that data is If there is no title of that data found then a -1 will be in that 
#   spot representing no index
#       index 0 is Plasma Frequency
#       index 1 is Collision Frequency
#       index 2 is x
#       index 3 is y
#       index 4 is z
# OUTPUTS: 
#   x: a meshgrid of the x distance
#   y: a meshgrid of the y distance
#   WPhz: a meshgrid of the plasma frequency
#   nu: a meshgrid of the collision frequency
# def dataHandling2D(datalist,listOfColumns): 
#     xline = [0]
#     yline = [0]
#     WPhzline = [1]
#     nuline = [1]
#     count = 1
#     for i in range(1,len(datalist)):
#         if datalist[i,0] == 
#         count+=1
#     for i in range(0,len(datalist)):
#         for k in range(0,len(datalist[i])):
#             if k == listOfColumns[0]:
#                 xline.append(datalist[i,k])
#             elif k == listOfColumns[2]:
#                 xline.append(datalist[i,k])
#     x = np.outer(xline,np.ones(len(xline)))
#     x = x.copy().T
#     y = np.outer(yline,np.ones(len(xline)))
#     return 

#exact
# this function calculates the exact values to be used for graphing, for 
# comparison reasons
# INPUTS: 
#   kstart
#   kend
#   ddx
#   c_0
#   wpHz_plasma
#   nuHz_plasma
# OUTPUTS: 
#   wHz_exact
#   magR_exact
#   magT_exact
def exact(kstart,kend,ddx,c_0,wpHz_plasma,nuHz_plasma):
    d = (kend-kstart)*ddx
    wHz_exact = np.linspace(0.01, 1,10000)*(10**10) #4e9
    w = wHz_exact*2*math.pi #2.5132e10
    lam = c_0/wHz_exact #0.0119266
    kList = 2*math.pi/lam #526.81
    Op = wpHz_plasma*2*math.pi/w #1
    Oc = nuHz_plasma/w #0.039788
    Np = np.emath.sqrt(1-pow(Op,2)/(1-1j*Oc)) #0
    gamp = 1j*kList*Np #0
    T_exact = (4*Np*np.exp(-1j*kList*d*(Np-1)))/((Np+1)**2 - (Np-1)**2*np.exp(-2*gamp*d))
    R_exact = (1-(Np**2))*(1-np.exp(-2*gamp*d))/((Np+1)**2 - (Np-1)**2*np.exp(-2*gamp*d))
    magR_exact = 10*np.log10(abs(R_exact**2))
    magT_exact = 10*np.log10(abs(T_exact**2))
    return wHz_exact , magR_exact , magT_exact

#appending
# this function appends the itemsfrom the inputted list into a list with only    
# real numbers called col
# INPUTS: 
#   listEZ - list to be changing into real numbers and appened to the output
# OUTPUTS: 
#   col - a list of real numbers from the inputted list
def appending1D(listEZ):
    i = 0
    col = []
    for i in range(0, len(listEZ)):
        col.append(listEZ[i].real)
    return col
#appending2D
# this function appends the itemsfrom the inputted meshgrid into a meshgrid with only    
# real numbers 
# INPUTS: 
#   meshEZ - list to be changing into real numbers and appened to the output
#   lenght - the length of the meshgrid (assuming a square meshgrid)
# OUTPUTS: 
#   ezmesh - a meshgrid of real numbers from the inputted meshgrid
def appending2D(meshEZ,length):
    i = 0
    ezmesh = meshEZ.copy()
    for i in range(0, length):
        for k in range(0,length):
            ezmesh[i][k] = ezmesh[i][k].real
    return ezmesh
#WpNuSetup
# this function calculates the plasma and collision frequency from either the data 
# or create the plasma layer for testing
# INPUTS: 
#   dimentions: A variable num that represents 1 2 or 3D code being run
#   NOdata:The boolean value to tell if there was an data or not passed into the function
#   data: the 2D list of inputs
#   listOfColumns: a list for referencing the data
#   wpHz_plasma: the num for the wpHz of the plasma
#   nuHz_plasma: the num for the nu of the plasma
#   kstart: a 3 variable long list to represent the x,y,z start of the plasma
#   kend: a 3 variable long list to represent the x,y,z end of the plasma
#   KE: a 3 digit list of the Kx, Ky, Kz values
#   dd: a 3 digit list of the ddx, ddy, ddz values
# OUTPUTS: 
#   wpHz: a list of data the interpolated wpHz
#   nu: a list of data containing the interpolated nu
#   xdata: the space between the data points in wpHz and nu
def WpNuSetup(dimen,NOdata,data, listOfColumns, wpHz_plasma, nuHz_plasma, plasmastart,plasmaend,KE,dd):
    #set wp and nu to the values on previous lines with no data inputs
    wpHzdatalist = [1]*len(data)
    nudatalist = [1]*len(data)
    xdata = [1]*len(data)
    ydata = [1]*len(data)
    zdata = [1]*len(data)
    #WITHOUT DATA 
    if NOdata == True:
        #use a linearspace from 0 to 150 when no data is present
        XmaxPoints = 15000*dd[0]
        YmaxPoints = 15000*dd[1]
        ZmaxPoints = 15000*dd[2]
        xdata = np.linspace(0,XmaxPoints,len(data))  
        ydata = np.linspace(0,YmaxPoints,len(data)) 
        zdata = np.linspace(0,ZmaxPoints,len(data)) 
    else:
        for u in range(0,len(data)):
            xdata[u] = float(data[u][listOfColumns[2]])*(10**3)
            #WITH DATA
            nudatalist[u] = (float(data[u][listOfColumns[1]]))
            wpHzdatalist[u] = (1.602176565*(10**-19)*math.sqrt(1/(8.854*(10**-12)*9.10938215*(10**-31)))/2/math.pi*math.sqrt(float(data[u][listOfColumns[0]])))
            if dimen >= 2: 
                ydata[u] = (float(data[u][listOfColumns[3]])*(10**3))
                if dimen >= 3: 
                        zdata[u] = (float(data[u][listOfColumns[4]])*(10**3))
        
    # Interpolate US3D data to EM grid
    xint = np.linspace(0,int(KE[0]+1)*dd[0],round(int(KE[0]+1)),endpoint=False)
    # xint = []
    # i=0
    # while i<=max(xdata):
    #     xint.append(i)
    #     i+= dd[0]
    if dimen >= 2:
        yint = np.linspace(0,int(KE[1]+1)*dd[1],round(int(KE[1]+1)),endpoint=False)
    #     i=0
    #     yint = []
    #     while i<=max(ydata):
    #         yint.append(i)
    #         i+= dd[1]
    if dimen == 3:
        zint = np.linspace(0,int(KE[2]+1)*dd[2],round(int(KE[2]+1)),endpoint=False)
    #     i=0
    #     zint = []
    #     while i<=max(zdata):
    #         zint.append(i)
    #         i+= dd[2]
    print("Before interpolation")
    #formula for 1D interpolation  yi = y1 + (y2-y1)/(x2-x1)*(xi-x1)
    nudatax = interp1d(xdata,nudatalist,'linear',fill_value="extrapolate")
    wpHzdatax = interp1d(xdata,wpHzdatalist,'linear',fill_value="extrapolate")
    if dimen >= 2:
        nudatay = interp1d(ydata,nudatalist,'linear',fill_value="extrapolate")
        wpHzdatay = interp1d(ydata,wpHzdatalist,'linear',fill_value="extrapolate")
        if dimen == 3:
            nudataz = interp1d(zdata,nudatalist,'linear',fill_value="extrapolate")
            wpHzdataz = interp1d(zdata,wpHzdatalist,'linear',fill_value="extrapolate")
    print("Got past the interpolation")
    wpHzintx = []
    nuintx = []
    wpHzinty = []
    nuinty = []
    wpHzintz = []
    nuintz = []
    #Take the interpolated funcion and find the data points tat are not zero and append the 
    #wpHz and nu data otherwise append the interpolated function wpHzint1 & nuint1 at point n
    n=0
    while(n<=max(xdata)): # Loop to fix end conditions where 'chip' might give negative values
        if wpHzdatax(n) < 0:
            wpHzintx.append(wpHzdatax(1))    
        else:
            wpHzintx.append(wpHzdatax(n)) 
        if nudatax(n) < 0:
            nuintx.append(nudatax(1))
        else:
            nuintx.append(nudatax(n))
        n+=dd[0]
    if dimen >= 2:
        n=0
        while(n<=max(ydata)): # Loop to fix end conditions where 'chip' might give negative values
            if wpHzdatay(n) < 0:
                wpHzinty.append(wpHzdatay(1))    
            else:
                wpHzinty.append(wpHzdatay(n)) 
            if nudatay(n) < 0:
                nuinty.append(nudatay(1))
            else:
                nuinty.append(nudatay(n))
            n+=dd[1]
    if dimen == 3:
        n=0
        while(n<=max(zdata)): # Loop to fix end conditions where 'chip' might give negative values
            if wpHzdataz(n) < 0:
                wpHzintz.append(wpHzdataz(1))    
            else:
                wpHzintz.append(wpHzdatay(n)) 
            if nudatay(n) < 0:
                nuintz.append(nudataz(1))
            else:
                nuintz.append(nudataz(n))
            n+=dd[2]
    # Translate US3D data on EM grid to set up room for input, reflected, and 
    # transmitted signals.  Also, need to reverse data for incoming GPS signal.
    print("Replacing all values less than 1 with 1")
    wpHzx=[1]*int(KE[0])
    nux=[1]*int(KE[0])
    i=plasmastart[0]
    while i<=(plasmaend[0]-1):
        wpx = wpHzintx[round(i-(plasmastart[0])+1)]
        nuxx = nuintx[round(i-(plasmastart[0])+1)]
        if wpx < 1:
            wpHzx[int(i)] = 1
        else:
            wpHzx[int(i)] = wpx
        if nuxx < 1:
            nux[int(i)] = 1
        else:
            nux[int(i)] = nuxx
        #USE THESE WHILE DEBUGGING
        if NOdata == True:
            wpHzx[int(i)] = wpHz_plasma # Run constant conditions for exact solution check
            nux[int(i)] = nuHz_plasma
        i+=1
    if dimen >= 2:
        wpHzy=[1]*int(KE[1])
        nuy=[1]*int(KE[1])
        if plasmastart[1] != 0 and plasmaend[1] != 0:
            i=plasmastart[1]
            while i<=(plasmaend[1]-1):
                wpy = wpHzinty[round(i-(plasmastart[1])+1)]
                nuyy = nuinty[round(i-(plasmastart[1])+1)]
                if wpy < 1:
                    wpHzy[int(i)] = 1
                else:
                    wpHzy[int(i)] = wpy
                if nuyy < 1:
                    nuy[int(i)] = 1
                else:
                    nuy[int(i)] = nuyy
                #USE THESE WHILE DEBUGGING
                if NOdata == True:
                    wpHzy[int(i)] = wpHz_plasma # Run constant conditions for exact solution check
                    nuy[int(i)] = nuHz_plasma
                i+=1
    if dimen == 3:
        wpHzz=[1]*int(KE[2])
        nuz=[1]*int(KE[2])
        if plasmastart[2] != 0 and plasmaend[2] != 0:
            i=plasmastart[2]
            while i<=(plasmaend[2]-1):
                wpz = wpHzintz[round(i-(plasmastart[2])+1)]
                nuzz = nuintz[round(i-(plasmastart[2])+1)]
                if wpz < 1:
                    wpHzz[int(i)] = 1
                else:
                    wpHzz[int(i)] = wpz
                if nuyy < 1:
                    nuz[int(i)] = 1
                else:
                    nuz[int(i)] = nuzz
                #USE THESE WHILE DEBUGGING
                if NOdata == True:
                    wpHzz[int(i)] = wpHz_plasma # Run constant conditions for exact solution check
                    nuz[int(i)] = nuHz_plasma
                i+=1 
    if dimen == 1:
        return wpHzx, nux, xint, 0,0,0, 0,0,0
    if dimen == 2:
        return wpHzx, nux, xint, wpHzy, nuy, yint, 0,0,0
    if dimen == 3:
        return wpHzx, nux, xint, wpHzy, nuy, yint, wpHzz, nuz, zint

#MagYfield
# this function recalculates the magnetic field in the Y direction
# This utilizes parallel processing
# INPUTS: 
#   hy: previous magnetic field numbers
#   ez: the electric field component
#   dt: time step
#   mu_0: free space permeability variable
#   ddx: delta X steps in the x direction
# OUTPUTS: 
#   HY: the recalculated magnetic field
def MagYfield(dimen,hy,ez,dt,mu_0,ddx):
    
    return HY