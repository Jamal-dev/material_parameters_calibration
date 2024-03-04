from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread
import os
from src.main import main
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.visualize import Visualization,plot_losses

class TrainingWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()  # Signal for when training finishes
    update_output = QtCore.pyqtSignal(str)  # Signal for output updates
    plot_losses_signal = QtCore.pyqtSignal(tuple)  
    visualize_data_signal = QtCore.pyqtSignal(tuple)
    model_data_signal = QtCore.pyqtSignal(tuple)  

    def __init__(self, params):
        super().__init__()
        self.params = params  # Parameters to pass to the main function

    def run(self):
        # Unpack parameters for readability
        filePath, fiberDirection, trainSize, validationSize, batchSize, psi, learningRate, num_epochs, printFreq, patience, modelSave, plotLosses, plotStresses, plotPsi = self.params
        
        # Call the main training function with the parameters and a lambda for updates
        main(Path(filePath),
             np.array(fiberDirection).reshape(3,1),
             train_size=trainSize,
             val_size=validationSize,
             batch_size=batchSize,
             expression=psi,
             lr=learningRate,
             num_epochs=num_epochs,
             print_every=printFreq,
             patience=patience,
             save_models=modelSave,
             plot_losses_bool=plotLosses,
             plot_stresses=plotStresses,
             plot_psi=plotPsi,
             update_ui=lambda message: self.update_output.emit(message),
             plot_losses_singal = lambda train_losses, val_losses: self.plot_losses_signal.emit((train_losses,val_losses)),
             visualize_data_signal = lambda model, strain_lin: self.visualize_data_signal.emit((model,strain_lin)),
             model_data_signal = lambda model, strain_lin: self.model_data_signal.emit((model,strain_lin)))  

        self.finished.emit()  # Emit finished signal when done


class Ui_Widget(object):
    def __init__(self, result_window):
        self.resultWindow = result_window  # Store the passed result_window for later use
    def updateResultWindow(self, message):
        # Ensure this function runs in the GUI thread
        self.resultWindow.newTextSignal.emit(message)

    def setupUi(self, Widget):

        # Widget.setObjectName("Material parameters optimization")
        Widget.resize(800, 600)
        Widget.setWindowTitle("Material Parameters Calibration")  # Set the window title
        Widget.setWindowIcon(QtGui.QIcon('app_icon.ico'))  # Set the window icon

        # Set up the main layout
        self.mainLayout = QtWidgets.QVBoxLayout(Widget)

        # Set up the form layout
        self.formLayout = QtWidgets.QFormLayout()

        # File Path Widgets
        self.lblFilePath = QtWidgets.QLabel("File Path")
        self.txtFilePath = QtWidgets.QLineEdit()
        self.btnFilePath = QtWidgets.QPushButton("Browse...")
        self.filePathLayout = QtWidgets.QHBoxLayout()  # Layout to hold the line edit and button
        self.filePathLayout.addWidget(self.txtFilePath)
        self.filePathLayout.addWidget(self.btnFilePath)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lblFilePath)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.FieldRole, self.filePathLayout)

        # Set up other widgets
        self.setupOtherWidgets()

        # Add the form layout to the main layout
        self.mainLayout.addLayout(self.formLayout)

        # Connections
        self.btnFilePath.clicked.connect(self.openFileNameDialog)

        # Connect the 'Start Train!' button to the onStartTrainClicked method
        self.pushButton.clicked.connect(self.onStartTrainClicked)
        

        # Set default values for the inputs
        self.setDefaultValues()

        self.retranslateUi(Widget)
        QtCore.QMetaObject.connectSlotsByName(Widget)

    def setupOtherWidgets(self):
        # Additional widgets as per your original code
        attributes = [
            ("lblPsi", "Ψ", "txtPsi", QtWidgets.QTextEdit()),
            ("lblFiberDirection", "Fiber Direction", "txtFiberDirection", QtWidgets.QLineEdit()),
            ("lblTrainSize", "Train size %", "txtTrainSize", QtWidgets.QLineEdit()),
            ("lblValSize", "Val size %", "txtValidationSize", QtWidgets.QLineEdit()),
            ("lblBatchSize", "Batch size", "txtBatchSize", QtWidgets.QLineEdit()),
            ("lblLr", "Learning rate", "txtLearningRate", QtWidgets.QLineEdit()),
            ("lblEpochs", "No of epochs", "txtNoEpochs", QtWidgets.QLineEdit()),
            ("lblPrintFreq", "Print Freq", "txtPrintFreq", QtWidgets.QLineEdit()),
            ("lblPatience", "Patience", "txtPatience", QtWidgets.QLineEdit()),
            # Add more widgets here as needed
        ]

        for i, (label_attr, label_text, line_edit_attr, line_edit_widget) in enumerate(attributes):
            setattr(self, label_attr, QtWidgets.QLabel(label_text))
            setattr(self, line_edit_attr, line_edit_widget)
            self.formLayout.setWidget(i + 1, QtWidgets.QFormLayout.LabelRole, getattr(self, label_attr))
            self.formLayout.setWidget(i + 1, QtWidgets.QFormLayout.FieldRole, getattr(self, line_edit_attr))

        # Additional controls like checkboxes and buttons
        self.setupAdditionalControls()

    def setupAdditionalControls(self):
        # Checkboxes and push button
        self.cbPlotPsi = QtWidgets.QCheckBox("Plot Ψ")
        self.cbModelSave = QtWidgets.QCheckBox("Model save")
        self.cbPlotStresses = QtWidgets.QCheckBox("Plot stresses")
        self.cbPlotLosses = QtWidgets.QCheckBox("Plot losses")
        self.pushButton = QtWidgets.QPushButton("Start Train!")
        # set as the default button
        self.pushButton.setShortcut("Return")  # Or "Enter"

        # Layout for checkboxes
        self.checkboxLayout1 = QtWidgets.QHBoxLayout()
        self.checkboxLayout1.addWidget(self.cbPlotPsi)
        self.checkboxLayout1.addWidget(self.cbModelSave)
        self.formLayout.setLayout(10, QtWidgets.QFormLayout.FieldRole, self.checkboxLayout1)

        self.checkboxLayout2 = QtWidgets.QHBoxLayout()
        self.checkboxLayout2.addWidget(self.cbPlotStresses)
        self.checkboxLayout2.addWidget(self.cbPlotLosses)
        self.formLayout.setLayout(11, QtWidgets.QFormLayout.FieldRole, self.checkboxLayout2)

        # Adding the start button
        self.formLayout.setWidget(12, QtWidgets.QFormLayout.FieldRole, self.pushButton)

    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            self.txtFilePath.setText(fileName)
    def plotLosses(self, data):
        (train_losses, val_losses) = data
        plot_losses(train_losses, val_losses)
    def plotStessesAndPsi(self, data):
        (model, strain_lin) = data
        if self.cbPlotStresses.isChecked():
            self.visualizer.plot_stresses(model,strain_lin)
        if self.cbPlotPsi.isChecked():
            self.visualizer.plot_psi(model,strain_lin)
    def visualizeData(self, data):
        (data_dict, indices_dict) = data
        self.visualizer = Visualization(data_dict, indices_dict)  # Initialize if not already

    def onStartTrainClicked(self):
        # Validate inputs first
        if not self.validateInputs():
            return  # Stop if validation fails

        # Fetch values if validation passes
        filePath = self.txtFilePath.text()
        trainSize = float(self.txtTrainSize.text())
        validationSize = float(self.txtValidationSize.text())
        batchSize = int(self.txtBatchSize.text())
        learningRate = float(self.txtLearningRate.text())
        noOfEpochs = int(self.txtNoEpochs.text())
        printFreq = int(self.txtPrintFreq.text())
        patience = int(self.txtPatience.text())
        psi = self.txtPsi.toPlainText()
        fiberDirection = eval(self.txtFiberDirection.text())

        # Fetch states of checkboxes
        plotPsi = self.cbPlotPsi.isChecked()
        modelSave = self.cbModelSave.isChecked()
        plotStresses = self.cbPlotStresses.isChecked()
        plotLosses = self.cbPlotLosses.isChecked()

        params = (filePath, 
                  fiberDirection, 
                  trainSize, 
                  validationSize, 
                  batchSize, 
                  psi, 
                  learningRate, 
                  noOfEpochs, 
                  printFreq, 
                  patience, 
                  modelSave, 
                  plotLosses, 
                  plotStresses, 
                  plotPsi)

        # Set up the worker and thread
        self.thread = QThread()
        self.worker = TrainingWorker(params)
        self.worker.plot_losses_signal.connect(self.plotLosses)
        self.worker.visualize_data_signal.connect(self.visualizeData)
        self.worker.model_data_signal.connect(self.plotStessesAndPsi)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.worker.update_output.connect(self.updateResultWindow)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Start the worker on the thread
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        # Print or use these values for further processing
        print(f"Starting training with parameters: File Path = {filePath}, Train Size = {trainSize}, etc...")
        # Add your training code or other processing here
    def setDefaultValues(self):
        # Set default values for inputs
        self.txtFilePath.setText('')
        self.txtPsi.setText('c1 * (J-1)**2 + c2 * (I1bar-3) + c3 * (I1bar-3)**2 + c4 * (I2bar-3) + c5 * (I2bar-3)**2')
        self.txtFiberDirection.setText('[1, 0, 0]')
        self.txtTrainSize.setText('0.88')
        self.txtValidationSize.setText('0.91')
        self.txtBatchSize.setText('32')
        self.txtLearningRate.setText('0.001')
        self.txtNoEpochs.setText('100')
        self.txtPrintFreq.setText('10')
        self.txtPatience.setText('10')
        # Set default states for checkboxes
        self.cbPlotPsi.setChecked(True)
        self.cbModelSave.setChecked(True)
        self.cbPlotStresses.setChecked(True)
        self.cbPlotLosses.setChecked(True)
    def validateInputs(self):
        # Validate inputs. Add your own validation rules here.
        # This is a basic example, adjust according to your requirements.
        if not self.txtFilePath.text().endswith('.csv'):
            QtWidgets.QMessageBox.warning(None, 'Input Error', 'File path must be a CSV file.')
            return False
        # check if train size and validation size are float
        if not self.txtTrainSize.text().replace('.', '', 1).isdigit():
            QtWidgets.QMessageBox.warning(None, 'Input Error', 'Train size must be a number.')
            return False
        if not self.txtValidationSize.text().replace('.', '', 1).isdigit():
            QtWidgets.QMessageBox.warning(None, 'Input Error', 'Validation size must be a number.')
            return False

        # Validate Fiber Direction as a vector
        try:
            # This will check if the fiber direction can be evaluated as a list and has three numeric components
            fiber_direction = eval(self.txtFiberDirection.text())
            if not (isinstance(fiber_direction, list) and len(fiber_direction) == 3 and all(isinstance(num, (int, float)) for num in fiber_direction)):
                raise ValueError  # This triggers the except block if the condition is false
        except:
            QtWidgets.QMessageBox.warning(None, 'Input Error', 'Fiber direction must be a numeric vector of length 3, e.g., [1, 0, 0].')
            return False

        if not self.txtBatchSize.text().isdigit():
            QtWidgets.QMessageBox.warning(None, 'Input Error', 'Batch size must be a number.')
            return False
        # check if batch size and Number of epochs are int
        if not self.txtNoEpochs.text().isdigit():
            QtWidgets.QMessageBox.warning(None, 'Input Error', 'Number of epochs must be a number.')
            return False
        if not self.txtPrintFreq.text().isdigit():
            QtWidgets.QMessageBox.warning(None, 'Input Error', 'Print frequency must be a number.')
            return False
        if not self.txtPatience.text().isdigit():
            QtWidgets.QMessageBox.warning(None, 'Input Error', 'Patience must be a number.')
            return False
        # check if learning rate is float
        if not self.txtLearningRate.text().replace('.', '', 1).isdigit():
            QtWidgets.QMessageBox.warning(None, 'Input Error', 'Learning rate must be a number.')
            return False
        # check if psi is empty or not
        if not self.txtPsi.toPlainText():
            QtWidgets.QMessageBox.warning(None, 'Input Error', 'Ψ cannot be empty.')
            return False
        
        # Check if the file exists
        if not os.path.exists(self.txtFilePath.text()):
            QtWidgets.QMessageBox.warning(None, 'File Error', 'The specified file does not exist.')
            return False
        return True
        
        
        # Add more validation checks as required
        # Return True if all checks pass
        return True


    def retranslateUi(self, Widget):
        _translate = QtCore.QCoreApplication.translate
        Widget.setWindowTitle(_translate("Widget", "Widget"))
        # Add any additional retranslate items here

class ResultWindow(QtWidgets.QDialog):
    # Define a new signal
    newTextSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(ResultWindow, self).__init__(parent)
        self.setWindowTitle('Training Log')
        self.setGeometry(750, 300, 500, 300)  # Adjust size and position as needed
        self.layout = QtWidgets.QVBoxLayout(self)

        self.resultTextEdit = QtWidgets.QTextEdit(self)
        self.resultTextEdit.setReadOnly(True)
        self.layout.addWidget(self.resultTextEdit)

        # Connect the signal to the appendText slot
        self.newTextSignal.connect(self.appendText)

    def appendText(self, text):
        # Ensure this function runs in the GUI thread
        self.resultTextEdit.moveCursor(QtGui.QTextCursor.End)
        self.resultTextEdit.insertPlainText(text + '\n')
        self.resultTextEdit.moveCursor(QtGui.QTextCursor.End)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # set icon
    app.setWindowIcon(QtGui.QIcon('app_icon.ico'))
    result_window = ResultWindow()
    mainWindow = QtWidgets.QWidget()
    ui = Ui_Widget(result_window=result_window)
    ui.setupUi(mainWindow)
    mainWindow.setWindowTitle("Material Parameters Calibration")
    mainWindow.setWindowIcon(QtGui.QIcon('app_icon.ico'))  

    

    # Create and show the result window
    
    mainWindow.show()
    result_window.show()



    sys.exit(app.exec_())
