import sys
from PyQt6.QtWidgets import QApplication
from gui import MainWindow

def main():
    """
    Main entry point for the Quantum Double-Slit Simulator application.
    Initializes the PyQt6 application loop and launches the MainWindow.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
