
from PyQt5.QtCore import QObject,pyqtSignal
from PyQt5 import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import time



class LassoWidget(FigureCanvas):
    selectionChanged = pyqtSignal(list)
    def __init__(self,parent,image=None):        
        self.fig = Figure(tight_layout=True)
        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
		
        self._ax = self.fig.gca()
        if image is not None:
            self._ax.imshow(image)
        line = {'color': 'green', 
        'linewidth': 2, 'alpha': 1}
        self._lasso = LassoSelector(self._ax, onselect=self.on_select,
                        lineprops=line, button=1)
        self.disconnect()
    
    def on_select(self,points):
        self.selectionChanged.emit(points)

    def disconnect(self):
        self._lasso.disconnect_events()
    
    def connect(self):
        self._lasso.connect_default_events()
    
    def clear(self):
        self._lasso.line.set_data([[],[]])
        self._lasso.line.set_visible(False)
        self.verts = None

    def updateImage(self,image):
        self._ax.clear()
        self._ax.imshow(image)
        self.fig.canvas.draw()
        #self.flush_events()
