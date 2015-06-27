import enaml
from enaml.qt.qt_application import QtApplication

if __name__ == "__main__":
    with enaml.imports():
        from controlsui.mk1.mockup import Main

    app = QtApplication()
    main_view = Main()
    main_view.show()
    main_view.redraw_timer.start()
    app.start()
