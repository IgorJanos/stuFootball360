//-----------------------------------------------------------------------------
//
//  Football360 Exporter
//
//  Author : Igor Janos
//
//-----------------------------------------------------------------------------
#include "pch.h"


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    setlocale(LC_NUMERIC, "C");

    Args        args;
    if (!args.parse(argc, argv)) {
        return -1;
    }


    if (args.split.size() == 2) {

        // Execute split
        return taskSplit(args);

    } else {

        // Execute export
        return taskExport(args);

    }

    MainWindow w;
    w.show();
    return a.exec();
}
