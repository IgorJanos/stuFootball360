//-----------------------------------------------------------------------------
//
//  Football360 Exporter
//
//  Author : Igor Janos
//
//-----------------------------------------------------------------------------
#include "pch.h"


static bool executeExport(
        QSharedPointer<Exporter::PipelineSource<Exporter::Image>> source,
        QSharedPointer<Exporter::CropRenderer> renderer,
        QSharedPointer<Exporter::DatasetSink> sink
    )
{

    bool isComplete = false;

    // Exporting process
    source->reset();
    while (!isComplete) {
        if (source->hasCurrent()) {

            auto inputImage = source->current();
            if (inputImage) {

                // TODO: random sampling
                Exporter::CropSample    crop;

                auto outputImage = renderer->render(inputImage, crop);
                sink->write(outputImage, crop);

                // Are we done ?
                if (sink->isComplete()) {
                    isComplete = true;
                }
            }

            // advance
            source->next();

        } else {
            isComplete = true;
        }
    }

    return true;
}



bool taskExport(Args &args)
{

    /*

      1. Load split JSON
      2. Load preset JSON
      3. Open output Context - store preset JSON inside
            4. Loop over chosen images

    */

    // Load split JSON
    auto        images = loadSplitJson(args.inputSplitJson.c_str());

    // Load preset JSON
    Preset      preset;
    preset.read(args.inputPresetJson.c_str());

    std::string     outputFile;
    QStringList     imageList;

    // Are we doing training or validation set ?
    if (args.outputValidationH5.empty()) {

        // We're training
        outputFile = args.outputTrainingH5;
        imageList = images.first;
    } else {

        // We're validating
        outputFile = args.outputValidationH5;
        imageList = images.second;
    }

    // How many images do we need ?
    int     nFiles = imageList.size();
    int     totalImages = preset.nImages;
    if (nFiles <= 0) {
        printf("Error: nFiles <= 0\n");
        return false;
    }

    // Cap the limit to 200
    int     perImage = ceil((float)totalImages / (float)nFiles);
    perImage = min(perImage, 200);
    int     cycles = ceil((float)totalImages / (float)(nFiles*perImage));


    //----------------------------------------------------
    //  Build the pipeline

    auto s1 = makeNew<Exporter::DatasetImageSource>(args.inputFolder.c_str(), imageList);
    auto s2 = makeNew<Exporter::Repeater>(s1, perImage);
    auto s3 = makeNew<Exporter::CycleCounter>(s2, cycles);


    QSurfaceFormat      glFormat;
    //glFormat.setVersion(3, 2);
    //glFormat.setProfile(QSurfaceFormat::CompatibilityProfile);
    //glFormat.setSamples(16);
    QSurfaceFormat::setDefaultFormat(glFormat);

    auto offs = makeNew<QOffscreenSurface>();
    offs->create();


    auto renderer = makeNew<Exporter::CropRenderer>(offs.get(), preset.renderSize);

    // TODO: params ...
    auto sink = makeNew<Exporter::DatasetSink>(
                outputFile.c_str(),
                preset.scaleSize,
                totalImages
            );


    printf("Starting export : %d images\n", totalImages);

    // Execute export !
    executeExport(s3, renderer, sink);

    printf("Export complete.\n");

    return true;
}
















