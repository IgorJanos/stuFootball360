//-----------------------------------------------------------------------------
//
//  Football360 Exporter
//
//  Author : Igor Janos
//
//-----------------------------------------------------------------------------
#include "pch.h"

#include <algorithm>
#include <random>




bool taskSplit(Args &args)
{
    // Sanity checks
    if (!dirExists(args.inputFolder.c_str())) {
        printf("Error: Input folder does not exist: %s\n",
               args.inputFolder.c_str()
               );
        return false;
    }

    /*
        1. Scan all files in folder
        2. Shuffle the list
        3. Split them
        4. Save split.JSON
    */

    // Get all files
    QStringList    allFiles;
    listDir(args.inputFolder.c_str(), "", allFiles);

    printf("Scanning : %s\n", args.inputFolder.c_str());
    printf("Found files : %d\n", (int)allFiles.size());

    // Shuffle them
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(allFiles), std::end(allFiles), rng);

    // Get the number of files
    int count = allFiles.size();
    int nTrain = int(ceil(count * args.split[0] / 100.0f));
    int nVal = int(ceil(count * args.split[1] / 100.0f));
    int i;
    nVal = min(nVal, count-nTrain);

    printf("   nTrain : %d\n", nTrain);
    printf("   nVal   : %d\n", nVal);

    // Store them in a JSON
    printf("\n");
    printf("Storing into : %s\n", args.outputSplitJson.c_str());
    QFile       file(args.outputSplitJson.c_str());
    if (file.open(QIODevice::WriteOnly)) {
        QJsonObject json;

        // Store the arrays
        QJsonArray  arrayTrain;
        for (i=0; i<nTrain; i++) {
            arrayTrain.append(allFiles[i]);
        }

        QJsonArray  arrayVal;
        for (i=0; i<nVal; i++) {
            arrayVal.append(allFiles[nTrain + i]);
        }

        json["train"] = arrayTrain;
        json["val"] = arrayVal;

        // Store into the file
        file.write(QJsonDocument(json).toJson());
    }

    return true;
}


