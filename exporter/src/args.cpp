//-----------------------------------------------------------------------------
//
//  Football360 Exporter
//
//  Author : Igor Janos
//
//-----------------------------------------------------------------------------
#include "pch.h"



//-----------------------------------------------------------------------------
//
//  Args class
//
//-----------------------------------------------------------------------------

/*

    -in <IMAGE_FOLDER>          = input folder
    -is <SPLIT_JSON>            = input split json file
    -ip <PRESET_JSON>           = input preset json file
    -split 90;10                = split percentages of the whole set
    -os <SPLIT_JSON>            = output split json file
    -ot <TRAINING_H5>           = output H5 file for training images
    -ov <VALIDATION_H5>         = output H5 file for validation images

*/

Args::Args()
{

}

bool Args::parse(int argc, char *argv[])
{
    int i = 1;
    while (i < argc) {

        if (strcmp(argv[i], "-in") == 0) {
            i ++;
            if (i >= argc) {
                printf("Expected input folder!!\n");
                return false;
            }
            this->inputFolder = std::string(argv[i]);
        } else
        if (strcmp(argv[i], "-is") == 0) {
            i ++;
            if (i >= argc) {
                printf("Expected input split json file!!\n");
                return false;
            }
            this->inputSplitJson = std::string(argv[i]);
        } else
        if (strcmp(argv[i], "-ip") == 0) {
            i ++;
            if (i >= argc) {
                printf("Expected input preset json file!!\n");
                return false;
            }
            this->inputPresetJson = std::string(argv[i]);
        } else
        if (strcmp(argv[i], "-split") == 0) {
            i ++;
            if (i >= argc) {
                printf("Expected list of split percentages!!\n");
                return false;
            }

            std::istringstream iss(argv[i]);
            std::string s;
            while (std::getline(iss, s, ';')) {
                float f = 0.0;
                if (sscanf(s.c_str(), "%f", &f) == 1) {
                    this->split.push_back(f);
                } else {
                    printf("Error: Cannot parse float value: %s !!\n", s.c_str());
                    return false;
                }
            }

            if (this->split.size() != 2) {
                printf("Error: Expected a list of 2 floats !!\n");
                return false;
            }

        } else
        if (strcmp(argv[i], "-os") == 0) {
            i ++;
            if (i >= argc) {
                printf("Expected output split json file!!\n");
                return false;
            }
            this->outputSplitJson = std::string(argv[i]);
        } else
        if (strcmp(argv[i], "-ot") == 0) {
            i ++;
            if (i >= argc) {
                printf("Expected output training images H5 file!!\n");
                return false;
            }
            this->outputTrainingH5 = std::string(argv[i]);
        } else
        if (strcmp(argv[i], "-ov") == 0) {
            i ++;
            if (i >= argc) {
                printf("Expected output validation images H5 file!!\n");
                return false;
            }
            this->outputValidationH5 = std::string(argv[i]);
        } else
        {
            // Unexpected argument !!
            printf("Unexpected argument: %s\n", argv[i]);
            return false;
        }
        i++;
    }

    return true;
}

