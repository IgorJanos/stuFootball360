//-----------------------------------------------------------------------------
//
//  Football360 Exporter
//
//  Author : Igor Janos
//
//-----------------------------------------------------------------------------
#ifndef HELPERS_H
#define HELPERS_H


void listDir(QString path, QString prefix, QStringList &result);

bool dirExists(QString path);

QPair<QStringList, QStringList> loadSplitJson(QString filename);



inline int min(int a, int b) { return (a < b ? a : b); }


template<class T, typename... Args> QSharedPointer<T> makeNew(Args... args) {
    return QSharedPointer<T>(new T(args...));
}


class Preset
{
public:

    QSize           renderSize;
    QSize           scaleSize;
    QString         compression;
    int             nImages;

    // View
    QPair<float, float>     rangePan;
    QPair<float, float>     rangeTilt;
    QPair<float, float>     rangeRoll;
    QPair<float, float>     rangeFOV;

    // Distortion
    QString                 distortion;
    QPair<float, float>     k1;
    float                   epsK2;

public:
    Preset();

    bool read(QString filename);
    bool read(const QJsonObject &json);
};





#endif // HELPERS_H
