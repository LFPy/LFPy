#!/usr/bin/env python
'''
This is a scripts allowing the user to specify the path to a morphology file,
and determine the rotation angles that will rotate the apical dendrite along the
vertical z-axis. If desired, it will create a .rot-file that LFPy will
automatically use to set the default rotation alongside the morphology.
'''

#import some stuff
import pylab as pl
import LFPy
import os

#plot will pop up by itself
pl.interactive(1)

'''
Define some functions for plotting
'''
def plot_linepiece(ax, cell, i, color):
    ax.plot([cell.xstart[i], cell.xend[i]],
        [cell.ystart[i], cell.yend[i]],
        [cell.zstart[i], cell.zend[i]], color = color, lw = cell.diam[i])


def plot_morpho_indices(cell, new_fig = True):
    from mpl_toolkits.mplot3d import Axes3D
    if new_fig:
        fig = pl.figure(figsize=[10, 10])
    else:
        fig = pl.gcf()
    ax = Axes3D(fig)
    if cell.get_idx(section='apic').size > 0:
        apic = cell.get_idx(section='apic')
    else:
        apic = cell.get_idx(section='dend')
    for i in range(apic.size):
        try:
            ax.text(cell.xmid[apic[i]],cell.ymid[apic[i]],cell.zmid[apic[i]], 
                str(apic[i]), fontsize=8)
        except TypeError:
            ax.text(cell.xmid[apic[i]],cell.ymid[apic[i]],cell.zmid[apic[i]], 
                str(apic[i]))
    
    apic = cell.get_idx(section='apic')
    soma = cell.get_idx(section='soma')
    axon = cell.get_idx(section='axon')
    dend = cell.get_idx(section='dend')
    
    for i in soma:
        plot_linepiece(ax, cell, i, 'k')
    for i in dend:
        plot_linepiece(ax, cell, i, 'g')
    for i in axon:
        plot_linepiece(ax, cell, i, 'grey')
    for i in apic:
        plot_linepiece(ax, cell, i, 'm')
    

    ax.set_xlabel('x [$\mu$m]')
    ax.set_ylabel('y [$\mu$m]')
    ax.set_zlabel('z [$\mu$m]')

    #lim_3d = max([max(abs(cell.xmid[apic])), \
    #                    max(abs(cell.ymid[apic])), \
    #                    max(abs(cell.zmid[apic]))])
    #lim_3d = round(lim_3d/2,-2)
    #
    #ax.set_xlim3d(-lim_3d,lim_3d)
    #ax.set_ylim3d(-lim_3d,lim_3d)
    #ax.set_zlim3d(-lim_3d,lim_3d)
    
    ax.axis(ax.axis('equal'))
    

def plot_morpho_3D_simple(cell, new_fig = True, color = 'k'):
    from mpl_toolkits.mplot3d import Axes3D
    if new_fig:
        fig = pl.figure(figsize=[10, 10])
    else:
        fig = pl.gcf()
    ax = Axes3D(fig)
    apic = cell.get_idx(section='apic')
    soma = cell.get_idx(section='soma')
    axon = cell.get_idx(section='axon')
    dend = cell.get_idx(section='dend')
    
    for i in soma:
        plot_linepiece(ax, cell, i, 'k')
    for i in dend:
        plot_linepiece(ax, cell, i, 'g')
    for i in axon:
        plot_linepiece(ax, cell, i, 'grey')
    for i in apic:
        plot_linepiece(ax, cell, i, 'm')

    ax.set_xlabel('x [$\mu$m]')
    ax.set_ylabel('y [$\mu$m]')
    ax.set_zlabel('z [$\mu$m]')

    lim_3d = max([max(abs(cell.xstart)), max(abs(cell.xend)), \
                        max(abs(cell.ystart)), max(abs(cell.yend)), \
                        max(abs(cell.zstart)), max(abs(cell.zend))])
    lim_3d = round(lim_3d,-2)

    ax.set_xlim3d(-lim_3d,lim_3d)
    ax.set_ylim3d(-lim_3d,lim_3d)
    ax.set_zlim3d(-lim_3d,lim_3d)

def det_rotationangles2(cell,segment1,segment2):
    '''return rotationangles around x- and y-axis, so the line between two
    chosen segments is aligned in parallell to the z-axis, corrected to soma.'''
    R = pl.sqrt((cell.xmid[segment1]-cell.xmid[segment2])**2 \
            + (cell.ymid[segment1]-cell.ymid[segment2])**2 \
            + (cell.zmid[segment1]-cell.zmid[segment2])**2)

    rot_x = pl.pi+pl.arctan2(cell.ymid[segment1]-cell.ymid[segment2],\
                                   cell.zmid[segment1]-cell.zmid[segment2])
    rot_y = -pl.pi+pl.arcsin((cell.xmid[segment1]-cell.xmid[segment2])/R)

    rotation = {
        'x' : rot_x,
        'y' : rot_y
        }
    return rotation


'''
MAIN SCRIPT
'''

pl.close('all')

morphologyPath = input('Enter path to morphology file: ')
print('You entered %s.' % morphologyPath)

cellparams = {
    'morphology' : morphologyPath,
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 10,
}

rotationFilePath = morphologyPath[:-4]+'.rot'

if os.path.isfile(rotationFilePath):
    print(('It appear as though there exist a file %s already' % rotationFilePath))
    print('I don\'t know what to do with all of this...')
    print('It\'s soo depressive....')
else:
    print('Will create file %s with the rotation information' % rotationFilePath)

    ok = 'n'

    while ok != 'y':

        cell = LFPy.Cell(**cellparams)

        plot_morpho_indices(cell)

        print('Identify TWO indices to be aligned with the z-axis in the plot')
        
        [a, b] = eval(input('Type them in here (sep. by comma): '))

        r_a = pl.sqrt(cell.xmid[a]**2 + cell.ymid[a]**2 + cell.zmid[a]**2)
        r_b = pl.sqrt(cell.xmid[b]**2 + cell.ymid[b]**2 + cell.zmid[b]**2)

        if r_a < r_b:
            b1 = a
            a1 = b
        else:
            a1 = a
            b1 = b

        rotation = det_rotationangles2(cell, segment1=a1,segment2=b1)
        print('Determined rotation angles to be (in radians): ', rotation)

        cell.set_rotation(**rotation)

        plot_morpho_3D_simple(cell)

        ok = input('Does it look correctly rotated (y)? ')
        print(ok)

    printtofile = input('Shall I write the file %s then (y) ? ' % rotationFilePath)
    if printtofile == 'y':
        F = file(rotationFilePath,'w')

        F.write('x=%g\n' % rotation['x'])
        F.write('y=%g\n' % rotation['y'])
        F.write('z=%g\n' % 0)

        F.close()
        
        print('file %s created!' % rotationFilePath)
