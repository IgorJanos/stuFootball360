//-----------------------------------------------------------------------------
//
//  Football360 Exporter
//
//  Author : Igor Janos
//
//-----------------------------------------------------------------------------
#include "pch.h"



bool dirExists(QString path)
{
    QDir dir(path);
    return dir.exists();
}



void listDir(QString path, QString prefix, QStringList &result)
{
    QDirIterator    it(path, QDir::Dirs | QDir::NoDotAndDotDot | QDir::Files);
    while (it.hasNext()) {
        QFileInfo       fi(it.next());
        if (fi.isDir()) {
            listDir(path+"/"+fi.fileName(), prefix+"/"+fi.fileName(), result);
        } else
        if (fi.isFile()) {

            // we only take JPG
            if (fi.fileName().indexOf(".jpg") >= 0) {
                result.append(prefix + "/" + fi.fileName());
            }
        }
    }
}


QPair<QStringList, QStringList> loadSplitJson(QString filename)
{
    QPair<QStringList, QStringList> result;

    QFile       file(filename);
    if (file.open(QIODevice::ReadOnly)) {
        QByteArray          fileData = file.readAll();
        QJsonParseError     err;
        QJsonDocument       json(QJsonDocument::fromJson(fileData, &err));

        // Skusime loadnut
        auto obj = json.object();
        if (obj.contains("train") && obj["train"].isArray()) {
            auto items = obj["train"].toArray();
            for (int i=0; i<items.size(); i++) {
                result.first.append(items[i].toString());
            }
        }
        if (obj.contains("val") && obj["val"].isArray()) {
            auto items = obj["val"].toArray();
            for (int i=0; i<items.size(); i++) {
                result.second.append(items[i].toString());
            }
        }
    }

    return result;
}

QList<float> readListFloat(const QJsonObject &json, QString name)
{
    QList<float> result;
    if (json.contains(name) && json[name].isArray()) {
        auto items = json[name].toArray();
        for (int i=0; i<items.size(); i++) {
            result.append(items[i].toDouble());
        }
    }
    return result;
}

QList<int> readListInt(const QJsonObject &json, QString name)
{
    QList<int> result;
    if (json.contains(name) && json[name].isArray()) {
        auto items = json[name].toArray();
        for (int i=0; i<items.size(); i++) {
            result.append(items[i].toInt());
        }
    }
    return result;
}

QPair<float, float> toPair(QList<float> alist)
{
    return QPair<float, float>(alist[0], alist[1]);
}

QSize toSize(QList<int> alist)
{
    return QSize(alist[0], alist[1]);
}

QString readString(const QJsonObject &json, QString name)
{
    if (json.contains(name) && json[name].isString()) {
        return json[name].toString();
    }
    return "";
}

int readInt(const QJsonObject &json, QString name)
{
    if (json.contains(name) && json[name].isDouble()) {
        return json[name].toInt();
    }
    return 0;

}

float readFloat(const QJsonObject &json, QString name)
{
    if (json.contains(name) && json[name].isDouble()) {
        return json[name].toDouble();
    }
    return 0;
}


//-----------------------------------------------------------------------------
//  Preset
//-----------------------------------------------------------------------------

Preset::Preset() :
    renderSize(1920, 1080),
    scaleSize(448, 448),
    compression("png"),
    nImages(1000),
    rangePan(-40, 40),
    rangeTilt(-25, -2),
    rangeRoll(-2, 2),
    rangeFOV(10, 50)
{
}

bool Preset::read(QString filename)
{
    QFile       file(filename);
    if (file.open(QIODevice::ReadOnly)) {

        rawData = file.readAll();

        QJsonParseError     err;
        QJsonDocument       json(QJsonDocument::fromJson(rawData, &err));

        return read(json.object());
    }

    return false;
}

bool Preset::read(const QJsonObject &json)
{
    renderSize = toSize(readListInt(json, "renderSize"));
    scaleSize = toSize(readListInt(json, "scaleSize"));
    compression = readString(json, "compression");
    nImages = readInt(json, "nImages");

    // View
    if (json.contains("view") && json["view"].isObject()) {
        auto view = json["view"].toObject();
        rangePan = toPair(readListFloat(view, "pan"));
        rangeTilt = toPair(readListFloat(view, "tilt"));
        rangeRoll = toPair(readListFloat(view, "roll"));
        rangeFOV = toPair(readListFloat(view, "fov"));
    }

    // Distortion
    distortion = readString(json, "distortion");
    if (json.contains("distortionParams") && json["distortionParams"].isObject()) {
        auto dp = json["distortionParams"].toObject();
        k1 = toPair(readListFloat(dp, "k1"));
        epsK2 = readFloat(dp, "epsK2");
    }

    return true;
}







//-----------------------------------------------------------------------------
//  Random helpers
//-----------------------------------------------------------------------------

std::default_random_engine		generator(
        std::chrono::system_clock::now().time_since_epoch().count()
        );

float uniform(QPair<float,float> args)
{
    std::uniform_real_distribution<float> d(args.first, args.second);
    return d(generator);
}

float normal(float mean, float stddev)
{
    std::normal_distribution<float> d(mean, stddev);
    return d(generator);
}












