QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets openglwidgets concurrent

QT_CONFIG -= no-pkg-config

CONFIG += c++11
CONFIG += precompile_header
CONFIG += link_pkgconfig

mac {
  PKG_CONFIG = /opt/homebrew/bin/pkg-config
}

PKGCONFIG += opencv4 hdf5

INCLUDEPATH += /Users/janos/.lib

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

PRECOMPILED_HEADER = pch.h

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    src/args.cpp \
    src/exporter.cpp \
    src/helpers.cpp \
    src/taskExport.cpp \
    src/taskSplit.cpp

HEADERS += \
    mainwindow.h \
    pch.h \
    src/args.h \
    src/exporter.h \
    src/helpers.h \
    src/indicators.h \
    src/tasks.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    shaders.qrc
