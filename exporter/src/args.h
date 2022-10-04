//-----------------------------------------------------------------------------
//
//  Football360 Exporter
//
//  Author : Igor Janos
//
//-----------------------------------------------------------------------------
#ifndef ARGS_H
#define ARGS_H


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


class Args
{
public:

    std::string         inputFolder;
    std::string         inputSplitJson;
    std::string         inputPresetJson;
    std::vector<float>  split;
    std::string         outputSplitJson;
    std::string         outputTrainingH5;
    std::string         outputValidationH5;

public:
    Args();

    bool parse(int argc, char *argv[]);

};



#endif // ARGS_H
