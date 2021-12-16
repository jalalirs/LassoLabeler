##### Globals #####

from PyQt5.QtWidgets import QApplication,QFileDialog,QMessageBox,QDialog
import PIL
import base64
import io

def _error(msg):
	msgBox = QMessageBox()
	msgBox.setIcon(QMessageBox.Critical)
	msgBox.setText(msg)  	
	msgBox.setWindowTitle("Alert")
	msgBox.setStandardButtons(QMessageBox.Close)
	retval = msgBox.exec_()

def _doOrNot(msg):
	qm = QMessageBox
	ret = qm.question(None,'', msg, qm.Yes | qm.No)
	return ret == qm.Yes

def notify(msg,ntype="error"):
	if ntype == "error":
		return _error(msg)
	elif ntype == "yesno":
		return _doOrNot(msg)
