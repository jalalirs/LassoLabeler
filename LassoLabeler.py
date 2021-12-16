#!/usr/bin/env python

''' A basic GUi to use ImageViewer class to show its functionalities and use cases. '''

from PyQt5 import QtCore, QtGui, uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QListWidgetItem
from PyQt5.QtGui import QPixmap,QIcon, QFontDatabase, QFont,QTextCursor,QPalette,QColor


import cv2
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import sys, os
import json
from dataset import Dataset
import signal
from utils import notify
from lassowidget import LassoWidget

DIR = os.path.dirname(os.path.realpath(__file__))
QLassoLabeler, Ui_LassoLabeler = uic.loadUiType(f"{DIR}/LassoLabeler.ui", resource_suffix='')

# for ctrl + c to kill
signal.signal(signal.SIGINT, signal.SIG_DFL)

class QLabelsQWidget (QtWidgets.QWidget):
    def __init__ (self, parent = None):
        super(QLabelsQWidget, self).__init__(parent)
        
        self.lblName  = QtWidgets.QLabel()
        self.lblName.setMaximumWidth(160)
        self.lblName.setFont(QFont('Arial', 14))


        # self.currentCountLabel  = QtWidgets.QLabel()
        # self.currentCountLabel.setText(str(0))
        # self.currentCountLabel.setMaximumWidth(30)
        # self.currentCountLabel.setFont(QFont('Arial', 14))

        self.iconQLabel = QtWidgets.QLabel()

        self.allQHBoxLayout  = QtWidgets.QHBoxLayout()
        self.allQHBoxLayout.addWidget(self.iconQLabel, 0)
        self.allQHBoxLayout.addWidget(self.lblName, 1)
        #self.allQHBoxLayout.addWidget(self.currentCountLabel, 2)
        self.setLayout(self.allQHBoxLayout)
        self.lblName.setStyleSheet('''color: rgb(0, 0, 0);''')

    def setName(self, text):
        self.lblName.setText(text)

    def setIcon(self, imagePath):
        img = QtGui.QPixmap(imagePath)
        img = img.scaledToWidth(64)
        self.iconQLabel.setPixmap(img)
    
    # def setCurrentCount(self,count):
    #      self.currentCountLabel.setText(str(count))

    def name(self):
        return self.lblName.text()

class LassoLabeler(QLassoLabeler, Ui_LassoLabeler):
    def __init__(self, parent=None):
        super(LassoLabeler,self).__init__(parent)

        self.setupUi(self)
        self._actualImageWidget = LassoWidget(self.image_widget)
        self.image_layout.addWidget(self._actualImageWidget)
        self._actualImageWidget.selectionChanged.connect(self.on_lasso_finished)

        self._maskWidget = LassoWidget(self.mask_widget)
        self.mask_layout.addWidget(self._maskWidget)
        self._maskWidget.disconnect()

        self._boundingboxWidget = LassoWidget(self.boundingbox_widget)
        self.boundingbox_layout.addWidget(self._boundingboxWidget)
        self._boundingboxWidget.disconnect()

        self.ls_contours.customContextMenuRequested[QtCore.QPoint].connect(self.on_contour_rightClicked)
        self.ls_keys.customContextMenuRequested[QtCore.QPoint].connect(self.on_keys_rightClicked)

        self.dataset = None
        self.storeBoundingBox = False
        self.keysWidget = {}
        self.applyStyle()

    def applyStyle(self):
        os.chdir(DIR + '/style')
        if QFontDatabase.addApplicationFont("fonts/ubuntu.ttf") != -1:
                font = QFont("Ubuntu")
                font.setPointSize(9)
                font.setStyleHint(QFont.SansSerif)
                QApplication.setFont(font)

        self.setProperty('css',True)
        for w in self.findChildren(QtWidgets.QWidget):
                w.setProperty('css',True)

        # hack: custom popup in combox
        for cbo in self.findChildren(QtWidgets.QComboBox):
            cbo.setMaxVisibleItems(8)
            cboview = QtWidgets.QListView(self)
            cboview.setProperty('css', True)
            cbo.setView(cboview)
            cbo.view().verticalScrollBar().setProperty('css',True)

        self.setStyleSheet(open('seepro.css').read())

        os.chdir(DIR)

    def clear_and_populate(self):
        self.ls_keys.clear()
        self.ls_images.clear()
        self.ls_contours.clear()
        self.ls_objects.clear()

        for k in self.dataset.keys():
            
            keyWidget = QLabelsQWidget()

            keyWidget.setName(k)
            keyWidget.setIcon(self.dataset.keyImage(k))
            #keyWidget.setCurrentCount(self.dataset.keyCount(k))
            
            keyListWidgetItem = QtWidgets.QListWidgetItem(self.ls_keys)
            keyListWidgetItem.setSizeHint(keyWidget.sizeHint())
            
            self.keysWidget[k] = keyListWidgetItem

            self.ls_keys.addItem(keyListWidgetItem)
            self.ls_keys.setItemWidget(keyListWidgetItem, keyWidget)
        
        for name in self.dataset.itemNames():
            self.ls_images.addItem(name)
        
        self.ls_images.setCurrentRow(0)
    
    def update_image(self):
        image = self.dataset.currentImage()
        self._actualImageWidget.clear()
        self._actualImageWidget.updateImage(image)

        maskImage = self.dataset.currentMaskImage()
        self._maskWidget.clear()
        self._maskWidget.updateImage(maskImage)

        boundingboxImage = self.dataset.currentBoundingboxImage()
        self._boundingboxWidget.clear()
        self._boundingboxWidget.updateImage(boundingboxImage)


    def on_lasso_finished(self,points):
        objectId = self.ls_objects.currentItem().text()
        label = "_".join(objectId.split("_")[:-1])
        self.dataset.addShape(label,"polygon",points,objectId)
        if self.storeBoundingBox:
            pass
        
        currentCount = self.ls_contours.count()
        x1,y1,x2,y2 = self.dataset.getContourBoundingBox(objectId,currentCount)
        self.ls_contours.addItem(f'{x1},{y1} {x2},{y2}')

        self.update_image()
        if self.mn_save_automatically.isChecked():
            self.dataset.save(self.mn_save_boundingbox.isChecked())
    
    def on_ln_search_key_textChanged(self):
        t = self.ln_search_key.text()
        ls = self.ls_keys
        if t.strip() == "":
            for index in range(self.ls_keys.count()):
                ls.setRowHidden(index, False)
            return
        for index in range(ls.count()):
            item = ls.item(index)
            itemWidget = self.ls_keys.itemWidget(item)
            name = itemWidget.name()
            if t not in name:
                ls.setRowHidden(index, True)
            else:
                ls.setRowHidden(index, False)
  
    @QtCore.pyqtSlot(QListWidgetItem,QListWidgetItem)
    def on_ls_images_currentItemChanged(self,current,previous):
        
        if current is None:
            return
        
        if previous is not None and self.dataset.didChange() and not self.mn_save_automatically.isChecked():
            save = notify("Do you want save the current changes?","yesno")
            if save:
                self.dataset.save(self.mn_save_boundingbox.isChecked())


        currentName = current.text()

        # changing the image
        self.dataset.changeItem(currentName,False)
        self.update_image()

        # clearing lists
        self.ls_contours.clear()
        self.ls_objects.clear()

        # populating objects
        for o in self.dataset.objectNames():
            self.ls_objects.addItem(o)
        
    @QtCore.pyqtSlot(QListWidgetItem,QListWidgetItem)
    def on_ls_objects_currentItemChanged(self,current,previous):
        if current == None:
            self._actualImageWidget.disconnect()
            return
        
        self._actualImageWidget.connect()

        currentObject = current.text()
        self.ls_contours.clear()
    
        for i in range(self.dataset.shapesForObject(currentObject)):
            x1,y1,x2,y2 = self.dataset.getContourBoundingBox(currentObject,i)
            self.ls_contours.addItem(f'{x1},{y1} {x2},{y2}')
    
    @QtCore.pyqtSlot(QListWidgetItem,QListWidgetItem)
    def on_ls_contours_currentItemChanged(self,current,previous):
        if current is None:
            return
        currentContour = self.ls_contours.currentRow()
        currentObject = self.ls_objects.currentItem().text()
        # to prevent instant update after item is removed
        if currentContour > self.dataset.shapesForObject(currentObject) - 1:
            return
        
        self.dataset.fillInContour(currentObject,currentContour)
        self.update_image()
             
    @QtCore.pyqtSlot()
    def on_mn_load_dataset_triggered(self):
        folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        if not folder:
            QtWidgets.QMessageBox.warning(self, 'No Folder Selected', 'Please select a valid Folder')
            return
        sucess,dataset,errorMsg = Dataset.load(folder)
        if not sucess:
            notify(errorMsg)
            return
        self.dataset = dataset
        self.clear_and_populate()

    def on_keys_rightClicked(self,QPos):
        self.listMenu= QtWidgets.QMenu()
        menu_item = self.listMenu.addAction(QtWidgets.QAction('Create object',self,triggered=self.on_create_object_clicked))
        
        parentPosition = self.ls_keys.mapToGlobal(QtCore.QPoint(0, 0))        
        self.listMenu.move(parentPosition + QPos)
        self.listMenu.show()
    
    def on_create_object_clicked(self):
        # name, done = QtWidgets.QInputDialog.getText(self, 'Input Dialog', 'Enter the object name')
        # while done and name in self.dataset.objectNames():
        #     notify(f"{name} already exists!")
        #     name, done = QtWidgets.QInputDialog.getText(self, 'Input Dialog', 'Enter the object name')
        # if not done:
        #     return

        labelListItem = self.ls_keys.currentItem()
        labelWidget = self.ls_keys.itemWidget(labelListItem)
        label = labelWidget.name()

        name = self.dataset.createObject(label)

        self.ls_objects.addItem(name)
        self.ls_objects.setCurrentRow(self.ls_objects.count()-1)
            
    def on_contour_rightClicked(self,QPos):
        self.listMenu= QtWidgets.QMenu()
        menu_item = self.listMenu.addAction(QtWidgets.QAction('Remove contour',self,triggered=self.on_remove_contour_clicked))
        
        parentPosition = self.ls_contours.mapToGlobal(QtCore.QPoint(0, 0))        
        self.listMenu.move(parentPosition + QPos)
        self.listMenu.show()

    def on_remove_contour_clicked(self):
        currentContour = self.ls_contours.currentRow()
        currentObject = self.ls_objects.currentItem().text()
        
        self.dataset.deleteContour(currentObject,currentContour)
        self.ls_contours.takeItem(self.ls_contours.currentRow())
        
        self.ls_contours.selectionModel().clear()

        self.update_image()
        if self.mn_save_automatically.isChecked():
            self.dataset.save(self.mn_save_boundingbox.isChecked())


if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	form = LassoLabeler(None)
	form.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
    main()

    