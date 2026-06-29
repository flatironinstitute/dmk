# Alex Barnett's collection of viz tools for python matplotlib.
# mostly found online. 9/5/18

# todo:
# make draggablecolorbar work with ion()
# make a button to return zoompan to default (full image)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    """3d arrow. see:
    https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

        
class ZoomPan:
    """Direct zoom (mousewheel) and pan (left button mouse drag), without
    having to click on the toolbar. Recommend toolbar is removed to use.
    
    By seedoodude in this thread:
    https://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel
    """
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None
        
    def zoom_factory(self, ax, base_scale = 0.9):
        """base_scale appears to set the scale factor for each mousewheel click
        I made it <1 to reverse the zoom (Barnett 9/5/18)
        """
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion

def demo_ZoomPan():
    """demo the ZoomPan.
    Also remove the toolbar since it's overridden by ZoomPan.
    Barnett 9/5/18
    """
    n=256
    a = np.random.randn(n,n)
    pl.ion()
    # kill future toolbars (set to 'toolbar2' to get back) :
    mpl.rcParams['toolbar'] = 'None'
    fig,ax = pl.subplots()
    im = pl.imshow(a,cmap='jet')
    zp = ZoomPan()
    zp.zoom_factory(ax)
    zp.pan_factory(ax)
    pl.show()


def goodcolorbar(mappable):
    """colorbar that matches height of image. usage: goodcolorbar(imagehandle)

    From:
    https://joseph-long.com/writing/colorbars/
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


class DraggableColorbar(object):
    """colorbar that can be dragged to change image colormap vmin,vmax

    From:
    http://www.ster.kuleuven.be/~pieterd/python/html/plotting/interactive_colorbar.html
    Although, in 2014 Pieter DeGroot left academia for a company: materialise.
    """
    def __init__(self, cbar, mappable):
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        self.cycle = sorted([i for i in dir(mpl.cm) if hasattr(getattr(mpl.cm,i),'N')])
        self.index = self.cycle.index(cbar.get_cmap().name)

    def connect(self):
        """connect to all the events we need"""
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.keypress = self.cbar.patch.figure.canvas.mpl_connect(
            'key_press_event', self.key_press)

    def on_press(self, event):
        """on button press we will see if the mouse is over us and store some data"""
        if event.inaxes != self.cbar.ax: return
        self.press = event.x, event.y

    def key_press(self, event):
        if event.key=='down':
            self.index += 1
        elif event.key=='up':
            self.index -= 1
        if self.index<0:
            self.index = len(self.cycle)
        elif self.index>=len(self.cycle):
            self.index = 0
        cmap = self.cycle[self.index]
        self.cbar.set_cmap(cmap)
        self.cbar.draw_all()
        self.mappable.set_cmap(cmap)
        self.mappable.get_axes().set_title(cmap)
        self.cbar.patch.figure.canvas.draw()

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.cbar.ax: return
        xprev, yprev = self.press
        dx = event.x - xprev
        dy = event.y - yprev
        self.press = event.x,event.y
        #print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f'%(x0, xpress, event.xdata, dx, x0+dx))
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button==1:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax -= (perc*scale)*np.sign(dy)
        elif event.button==3:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax += (perc*scale)*np.sign(dy)
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()


    def on_release(self, event):
        """on release we reset the press data"""
        self.press = None
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def disconnect(self):
        """disconnect all the stored connection ids"""
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidpress)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidrelease)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidmotion)

def myshow(ax,im=None,fig=None):
    """my preferred interactive image shower, in matplotlib
    ax = axes handle
    im (optional) = image handle, provides interactive colorbar
    fig = figure handle (needed if im provided).
    Barnett 9/6/18
    """
    zp = ZoomPan()
    zp.zoom_factory(ax)
    zp.pan_factory(ax)
    if im!=None:
        cbar = fig.colorbar(im,aspect=8)
        cbar = DraggableColorbar(cbar,im)
        cbar.connect()
    pl.show()

def demo_myshow():
    """demo myshow
    """
    n=256
    a = np.random.randn(n,n)
    #pl.ion()     # seems to stop cbar dragging working!
    fig,ax = pl.subplots()
    im = pl.imshow(a,cmap='jet')
    myshow(ax,im,fig)
    #myshow(ax)
        
def demo_DraggableColorbar():
    """demo the DraggableColorbar. Barnett 9/5/18
    """
    n=256
    a = np.random.randn(n,n)
    #pl.ion()     # seems to stop dragging working!
    fig,ax = pl.subplots()
    im = pl.imshow(a,cmap='jet')
    cbar = fig.colorbar(im,aspect=8)
    cbar = DraggableColorbar(cbar,im)
    cbar.connect()
    pl.show()
    
def demo_ZoomPanColorbar():
    """demo the ZoomPan and DraggableColoabar together.
    Also remove the toolbar since it's overridden by ZoomPan.
    Barnett 9/5/18
    """
    n=256
    a = np.random.randn(n,n)
    #pl.ion()
    # kill future toolbars (set to 'toolbar2' to get back) :
    mpl.rcParams['toolbar'] = 'None'
    fig,ax = pl.subplots()
    im = pl.imshow(a,cmap='jet')
    cbar = fig.colorbar(im,aspect=8)
    cbar = DraggableColorbar(cbar,im)
    cbar.connect()
    zp = ZoomPan()
    zp.zoom_factory(ax)
    zp.pan_factory(ax)
    pl.show()
