//-----------------------------------------------------------------------------
//
//  Football360 Exporter
//
//  Author : Igor Janos
//
//-----------------------------------------------------------------------------
#include "pch.h"

#include <QImageReader>


namespace Exporter {



double toRad(double degrees)
{
    return degrees * M_PI / 180.0;
}

cv::Mat RotationMatrix(double rx, double ry, double rz)
{
    cv::Mat R_x = (cv::Mat_<double>(3,3) <<
            1,       0,         0,
            0,       cos(rx),   -sin(rx),
            0,       sin(rx),   cos(rx)
    );

    cv::Mat R_y = (cv::Mat_<double>(3,3) <<
            cos(ry),    0,      sin(ry),
            0,          1,      0,
            -sin(ry),   0,      cos(ry)
    );

    cv::Mat R_z = (cv::Mat_<double>(3,3) <<
            cos(rz),    -sin(rz),      0,
            sin(rz),    cos(rz),       0,
            0,          0,             1);

    cv::Mat R = R_z * R_y * R_x;
    return R;
}




//-----------------------------------------------------------------------------
//
//  DatasetImageSource
//
//-----------------------------------------------------------------------------

DatasetImageSource::DatasetImageSource(
        QString apath, QStringList aimages
        ) :
    path(apath),
    images(aimages),
    index(-1)
{

    QImageReader::setAllocationLimit(512 * 1024*1024);

}


QSharedPointer<Image> DatasetImageSource::current()
{
    QMutexLocker    l(&lock);

    if (index >= images.size()) return nullptr;

    QSharedPointer<Image>   result = QSharedPointer<Image>(new Image());

    // load the image file
    result->filename = path + images[index];
    result->image.load(result->filename);

    return result;
}

void DatasetImageSource::reset()
{
    QMutexLocker    l(&lock);
    index = 0;
}

void DatasetImageSource::next()
{
    QMutexLocker    l(&lock);

    if (index < images.size()) {
        index += 1;
    }
}

bool DatasetImageSource::hasCurrent()
{
    QMutexLocker    l(&lock);

    return (index >= 0 && index < images.size());
}


//-----------------------------------------------------------------------------
//
//  CycleCounter
//
//-----------------------------------------------------------------------------

CycleCounter::CycleCounter(
        QSharedPointer<PipelineSource<Image>> asrc, int acount
        ) :
    source(asrc),
    count(acount),
    index(-1)
{
}

QSharedPointer<Image> CycleCounter::current()
{
    QMutexLocker    l(&lock);

    if (index >= count) return nullptr;

    return source->current();
}

void CycleCounter::reset()
{
    QMutexLocker    l(&lock);

    source->reset();
    index = 0;

    // reset when end was reached
    while (index < count) {
        if (!source->hasCurrent()) {
            source->reset();
            index += 1;
        } else {
            break;
        }
    }
}

void CycleCounter::next()
{
    QMutexLocker    l(&lock);

    if (index >= count) return ;

    // delegate
    source->next();

    // reset when end was reached
    while (index < count) {
        if (!source->hasCurrent()) {
            source->reset();
            index += 1;
        } else {
            break;
        }
    }
}

bool CycleCounter::hasCurrent()
{
    QMutexLocker    l(&lock);

    if (index >= count) return false;

    // delegate
    return source->hasCurrent();
}


//-----------------------------------------------------------------------------
//
//  Repeater
//
//-----------------------------------------------------------------------------

Repeater::Repeater(
        QSharedPointer<PipelineSource<Image>> asrc, int acount
        ) :
    source(asrc),
    count(acount),
    index(0)
{


}

QSharedPointer<Image> Repeater::current()
{
    QMutexLocker    l(&lock);

    if (!cache && index == 0) {
        cache = source->current();
    }

    return cache;
}

void Repeater::reset()
{
    QMutexLocker    l(&lock);

    source->reset();
    index = 0;
    cache = nullptr;
}

void Repeater::next()
{
    QMutexLocker    l(&lock);

    index ++;
    if (index >= count) {
        cache = nullptr;
        index = 0;
        source->next();
    }
}

bool Repeater::hasCurrent()
{
    QMutexLocker    l(&lock);

    if (index > 0 && index < count) {
        return (cache ? true : false);
    }

    return source->hasCurrent();
}


//-----------------------------------------------------------------------------
//
//  CropSample
//
//-----------------------------------------------------------------------------


CropSample::CropSample() :
    p(0), t(0), r(0), fov(45),
    k1(0), k2(0)
{
}

CropSample::CropSample(const CropSample &av):
    p(av.p), t(av.t), r(av.r), fov(av.fov),
    k1(av.k1), k2(av.k2)
{

}

CropSample &CropSample::operator =(CropSample v)
{
    p = v.p; t = v.t; r = v.r; fov = v.fov;
    k1 = v.k1; k2 = v.k2;
    return *this;
}



//-----------------------------------------------------------------------------
//
//  InterpolatedFunction class
//
//-----------------------------------------------------------------------------

InterpolatedFunction::InterpolatedFunction() :
    invertible(true)
{

}

void InterpolatedFunction::reset()
{
    invertible = true;
    data.clear();
    xMin = xMax = yMin = yMax = 0.0;
}

void InterpolatedFunction::add(float x, float y)
{
    if (data.size() == 0) {

        data.push_back(std::make_pair(x,y));

        xMin = x;
        xMax = x;
        yMin = y;
        yMax = y;

    } else {

        // prikladame len pokial je funkcia prosta
        if (invertible) {
            if (y > yMax) {
                data.push_back(std::make_pair(x,y));
                xMin = (x < xMin ? x : xMin);
                xMax = (x > xMax ? x : xMax);
                yMin = (y < yMin ? y : yMin);
                yMax = (y > yMax ? y : yMax);
            } else {
                invertible = false;
            }
        }
    }
}

std::pair<float,float> InterpolatedFunction::xRange()
{
    return std::make_pair(xMin, xMax);
}

std::pair<float,float> InterpolatedFunction::yRange()
{
    return std::make_pair(yMin, yMax);
}


float InterpolatedFunction::getY(float x)
{
    int n = data.size();
    if (x >= xMin && x <= xMax && n >= 2) {
        for (int i=0; i<n-1; i++) {
            std::pair<float,float>  &cur = data[i];
            std::pair<float,float>  &next = data[i+1];
            if (x >= cur.first && x <= next.first) {
                float relative = (x - cur.first) / (next.first - cur.first);
                float y = cur.second + relative*(next.second - cur.second);
                return y;
            }
        }
    }

    return 0;
}

float InterpolatedFunction::getX(float y)
{
    int n = data.size();
    if (y >= yMin && y <= yMax && n >= 2) {
        for (int i=0; i<n-1; i++) {
            std::pair<float,float>  &cur = data[i];
            std::pair<float,float>  &next = data[i+1];
            if (y >= cur.second && y <= next.second) {
                float relative = (y - cur.second) / (next.second - cur.second);
                float x = cur.first + relative*(next.first - cur.first);
                return x;
            }
        }
    }

    return 0;
}



//-----------------------------------------------------------------------------
//
//  DatasetSink
//
//-----------------------------------------------------------------------------

DatasetSink::DatasetSink(
        const char *afilename,
        Preset *apreset
        ) :
    totalCount(apreset->nImages),
    writtenCount(0),
    scaleSize(apreset->scaleSize),
    compression(apreset->compression)
{
    // Open the file & make folder for images
    file = makeNew<H5::H5File>(afilename, H5F_ACC_TRUNC);
    groupImages = makeNew<H5::Group>(file->createGroup("images"));

    // Fill in INFO
    hsize_t         dims[1] = { (hsize_t)apreset->rawData.size() };
    H5::DataSpace   dspace(1, dims);
    H5::DataSet     dset = file->createDataSet(
                        "info", H5::PredType::NATIVE_UINT8, dspace
                        );
    dset.write(apreset->rawData.data(), H5::PredType::NATIVE_UINT8);
    dset.close();

    // Alloc data
    labelsData.reserve(2*apreset->nImages);
}

DatasetSink::~DatasetSink()
{
    if (file && writtenCount > 0) {
        // Write the labels data

        hsize_t         dims[2] = { (hsize_t)writtenCount, 2 };
        H5::DataSpace   lspace(2, dims);
        H5::DataSet     lset = file->createDataSet(
                                    "labels", H5::PredType::NATIVE_FLOAT, lspace
                                );
        lset.write(labelsData.data(), H5::PredType::NATIVE_FLOAT);
        lset.close();
    }

    if (groupImages) {
        groupImages->close();
    }
}

bool DatasetSink::isComplete()
{
    return (writtenCount >= totalCount);
}


static cv::Mat toMat(QImage const &img, int format)
{
   //same as convert mat to qimage, the fifth parameter bytesPerLine()
   //indicate how many bytes per row
   //If you want to copy the data you need to call clone(), else QImage
   //cv::Mat will share the buffer
   return cv::Mat(
               img.height(), img.width(), format,
               const_cast<uchar*>(img.bits()),
               img.bytesPerLine()
               ).clone();
}


void DatasetSink::write(QSharedPointer<RenderedImage> frame, CropSample sample)
{
    if (writtenCount >= totalCount) return ;

    // Rescale & compress
    cv::Mat     mFrame = toMat(frame->image, CV_8UC3);
    cv::Mat     mConv, mFinal;

    cv::cvtColor(mFrame, mConv, cv::COLOR_BGR2RGB);

    int         h = mConv.cols;
    int         w = mConv.rows;
    if (scaleSize.width() != w || scaleSize.height() != h) {
        cv::resize(
                mConv, mFinal,
                cv::Size(scaleSize.width(), scaleSize.height()),
                0, 0, cv::INTER_AREA
            );
    } else {
        mFinal = mConv;
    }

    // Append labels data
    labelsData.push_back(sample.k1);
    labelsData.push_back(sample.k2);

    // Compress
    std::vector<uchar>     data(1024*1024);
    if (compression == "jpg") {
        std::vector<int>        params;
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(100);

        cv::imencode(".jpg", mFinal, data, params);
    } else
    if (compression == "png") {
        cv::imencode(".png", mFinal, data);
    }

    // Store the file
    if (groupImages) {
        int dataSize = data.size();
        std::string name = std::to_string(writtenCount);

        hsize_t         dims[1] = { (hsize_t)dataSize };
        H5::DataSpace   dspace(1, dims);
        H5::DataSet     dset = groupImages->createDataSet(
                            name.c_str(), H5::PredType::NATIVE_UINT8, dspace
                            );
        dset.write(data.data(), H5::PredType::NATIVE_UINT8);
        dset.close();
    }

    writtenCount ++;
}



//-----------------------------------------------------------------------------
//
//  PinholeProgram
//
//-----------------------------------------------------------------------------

PinholeProgram::PinholeProgram(QOpenGLFunctions *func) :
    f(func),
    program(nullptr),
    posVertex(0),
    posTex(0),
    posTexture(0),
    posDistortTexture(0),
    posCanvas(0),
    posRK(0),
    posArgs(0),
    posK(0)
{
    heightWise = true;
}

void PinholeProgram::init()
{
    // Setup program
    program = new QOpenGLShaderProgram();
    program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/default.vert");
    program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/default.frag");
    program->link();

    posVertex = program->attributeLocation("vertex");
    posTex = program->attributeLocation("tex");
    posTexture = program->uniformLocation("texture");
    posDistortTexture = program->uniformLocation("textureDistort");
    posRK = program->uniformLocation("rk");
    posCanvas = program->uniformLocation("canvas");
    posArgs = program->uniformLocation("args");
    posK = program->uniformLocation("k");
}

void PinholeProgram::destroy()
{
    if (program) {
        delete program;
        program = nullptr;
    }
}

void PinholeProgram::bind()
{
    program->bind();
}

void PinholeProgram::unbind()
{

    program->release();
}

void PinholeProgram::setCanvas(QVector2D value)
{
    program->setUniformValue(posCanvas, value);
}

void PinholeProgram::setTexture(GLint value)
{
    program->setUniformValue(posTexture, value);
}

void PinholeProgram::setDistortTexture(GLint value)
{
    program->setUniformValue(posDistortTexture, value);
}

void PinholeProgram::setRK(QMatrix3x3 value)
{
    program->setUniformValue(posRK, value);
}

void PinholeProgram::setArgs(float gamma, QVector3D hsv)
{
    QVector4D   args(hsv, gamma);
    program->setUniformValue(posArgs, args);
}

void PinholeProgram::setK(QVector4D k)
{
    program->setUniformValue(posK, k);
}

void PinholeProgram::draw()
{
    // Vertices
    static const GLfloat vertices[] = {
        -1.0, 1.0,
         1.0, 1.0,
         1.0, -1.0,

        -1.0, 1.0,
        1.0, -1.0,
        -1.0, -1.0
    };
    static const GLfloat tex[] = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,

        0.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    };

    // Bind vertices
    f->glVertexAttribPointer(posVertex, 2, GL_FLOAT, GL_FALSE, 0, vertices);
    f->glVertexAttribPointer(posTex, 2, GL_FLOAT, GL_FALSE, 0, tex);
    f->glEnableVertexAttribArray(posVertex);
    f->glEnableVertexAttribArray(posTex);

    // Draw
    f->glDrawArrays(GL_TRIANGLES, 0, 6);

    // cleanup
    f->glDisableVertexAttribArray(posVertex);
    f->glDisableVertexAttribArray(posTex);
}


void PinholeProgram::prepareView(CropSample s, int width, int height)
{
    // Canvas
    QVector2D       _canvas(1.0, 1.0);
    if (heightWise) {
        if (height > 0) _canvas.setX((float)width / (float)height);
    } else {
        if (width > 0) _canvas.setY((float)height / (float)width);
    }


    setCanvas(_canvas);

    // Focal length
    double			f = 1;
    if (s.fov < 180) {
        f = 1.0 / (2.0 * tan(toRad(s.fov)/2.0));
    }

    // Camera Intrinsic, Rotation
    cv::Mat			K, R;
    R = RotationMatrix(toRad(s.t), toRad(s.p), toRad(s.r));
    K = (cv::Mat_<double>(3,3) <<
            f, 0, _canvas.x() / 2.0,
            0, f, _canvas.y() / 2.0,
            0, 0, 1
        );

    cv::Mat			RK = R * K.inv();
    RK.convertTo(RK, CV_32F);

    QMatrix3x3      _rk((const float*)RK.data);
    setRK(_rk);

}

cv::Mat PinholeProgram::getInverseRK(CropSample s, int width, int height)
{
    // Canvas
    QVector2D       _canvas(1.0, 1.0);
    if (heightWise) {
        if (height > 0) _canvas.setX((float)width / (float)height);
    } else {
        if (width > 0) _canvas.setY((float)height / (float)width);
    }

    // Focal length
    double			f = 1;
    if (s.fov < 180) {
        f = 1.0 / (2.0 * tan(toRad(s.fov)/2.0));
    }

    // Priprava matic
    cv::Mat			K, R;
    R = RotationMatrix(toRad(s.t), toRad(s.p), toRad(s.r));
    K = (cv::Mat_<double>(3,3) <<
            f, 0, _canvas.x() / 2.0,
            0, f, _canvas.y() / 2.0,
            0, 0, 1
        );

    cv::Mat			RK = R * K.inv();
    cv::Mat         RK_INV = RK.inv();

    return RK_INV;
}





//-----------------------------------------------------------------------------
//
//  CropRenderer
//
//-----------------------------------------------------------------------------


CropRenderer::CropRenderer(
        QOffscreenSurface *asurface,
        QSize asize
        ) :
    size(asize),
    context(nullptr),
    surface(asurface),
    program(nullptr),
    pixels(nullptr),
    texture(nullptr),
    texDistort(nullptr),
    isInitialized(false)
{
    // Naalokujeme data
    pixels = (uchar*)malloc(size.width() * size.height() * 3);
}

CropRenderer::~CropRenderer()
{
    qDebug() << "Renderer destroyed";

    if (context) {
        context->makeCurrent(surface);


        if (texture) {
            delete texture;
            texture = nullptr;
        }

        if (texDistort) {
            delete texDistort;
            texDistort = nullptr;
        }

        if (program) {
            program->destroy();
            delete program;
            program = nullptr;
        }

        context->doneCurrent();
        delete context;
        context = nullptr;
    }

    if (pixels) {
        free(pixels);
        pixels = nullptr;
    }
}

bool CropRenderer::initialize()
{

    // zrobime novy kontext
    context = new QOpenGLContext();
    context->setFormat(surface->format());
    context->create();
    context->makeCurrent(surface);
    context->extraFunctions()->initializeOpenGLFunctions();

    // Loadneme shaders
    program = new PinholeProgram(context->functions());
    program->init();

    auto f = context->functions();

    // Render Target
    f->glGenRenderbuffers(1, &rtTarget);
    f->glBindRenderbuffer(GL_RENDERBUFFER, rtTarget);
    f->glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, size.width(), size.height());
    f->glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // Depth Buffer
    f->glGenRenderbuffers(1, &dsTarget);
    f->glBindRenderbuffer(GL_RENDERBUFFER, dsTarget);
    f->glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, size.width(), size.height());
    f->glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // Framebuffer object
    f->glGenFramebuffers(1, &fbo);
    f->glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    f->glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rtTarget);
    f->glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, dsTarget);
    f->glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return true;
}


void CropRenderer::computeDistortTexture(InterpolatedFunction &func, float maxR, bool inverse)
{
    // Vypocitame maximalny diagonalny radius
    int n = 256;

    if (!texDistort) {
        texDistort = new QOpenGLTexture(QOpenGLTexture::Target1D);
        texDistort->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);

        texDistort->setSize(n);
        texDistort->setFormat(QOpenGLTexture::R32F);
        texDistort->allocateStorage(QOpenGLTexture::Red, QOpenGLTexture::Float32);
    }


    // Spocitame nove hodnoty
    std::vector<float>      f;
    f.resize(n);
    for (int i=0; i<n; i++) {
        double y = maxR * ((float)i / (float)(n-1));
        double r = 0;
        if (inverse) {
            r = func.getX(y);
        } else {
            r = func.getY(y);
        }
        if (y > 0) {
            f[i] = r/y;
        } else {
            f[i] = 1.0;
        }
    }

    texDistort->setData(QOpenGLTexture::Red, QOpenGLTexture::Float32, f.data());

}

static float distort(QVector4D k, float r)
{
    float r2 = r*r;
    float r4 = r2*r2;
    double d = 1.0 + k.x()*r2 + k.y()*r4 + k.z()*(r2*r4) + k.w()*(r4*r4);
    return d;
}


static float getMaxR(int width, int height)
{
    float aspect = (float)width / (float)height;
    float h = 0.5;
    float w = h * aspect;
    return sqrt(h*h + w*w);
}

QSharedPointer<RenderedImage> CropRenderer::render(
            QSharedPointer<Image> image,
            CropSample sample
        )
{

    if (!isInitialized) {
        isInitialized = initialize();
        // Error ?
        if (!isInitialized) return nullptr;
    }

    context->makeCurrent(surface);
    auto f = context->extraFunctions();

    //-----------------------------------------------
    //  View parameters & distortion function
    QVector4D       k(sample.k1, sample.k2, 0, 0);

    float maxR = getMaxR(size.width(), size.height());
    int n = 256;
    auto &_dist = dist;
    _dist.reset();
    for (int i=0; i<n; i++) {
        float r = 0.0 + 2.0*maxR*(float)i/(float)(n-1);
        float d = r * distort(k, r);
        _dist.add(r, d);
    }

    computeDistortTexture(_dist, maxR, true);


    //-----------------------------------------------
    //  Rendering
    f->glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    GLenum      bufs = GL_COLOR_ATTACHMENT0;
    f->glDrawBuffers(1, &bufs); //GL_COLOR_ATTACHMENT0);
    f->glClearColor(0.5f, 0.0f, 0.0f, 1.0f);
    f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    f->glViewport(0,0,size.width(),size.height());

    float       imw = 1.0;
    float       imh = 1.0;

    // Loadujeme obrazok
    if (lastImage != image) {
        lastImage = image;

        if (texture) {
            delete texture;
            texture = nullptr;
        }

        texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
        texture->setData(image->image, QOpenGLTexture::DontGenerateMipMaps);
        texture->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
        texture->setWrapMode(QOpenGLTexture::DirectionS, QOpenGLTexture::Repeat);
        texture->setWrapMode(QOpenGLTexture::DirectionT, QOpenGLTexture::Repeat);
    }

    // odlozime si rozlisko
    if (image) {
        imw = image->image.width();
        imh = image->image.height();
    }

    // Kreslime pohlad
    program->bind();
        // nahodime texturu
        if (texture) {
            texture->bind(0);
            program->setTexture(0);
        }
        if (texDistort) {
            texDistort->bind(2);
            program->setDistortTexture(2);
        }

        program->prepareView(sample, size.width(), size.height());
        program->setArgs(1.0, QVector3D(0.0, 1.0, 1.0));
        program->setK(k);
        program->draw();

    program->unbind();


    //----------------------------------------------
    //  Download result
    f->glReadBuffer(GL_COLOR_ATTACHMENT0);
    f->glReadPixels(0,0, size.width(), size.height(),
                 GLenum(GL_RGB), GLenum(GL_UNSIGNED_BYTE),
                 pixels
                 );

    // Clean up
    f->glBindRenderbuffer(GL_RENDERBUFFER, 0);
    f->glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Vratime vysledok
    auto result = makeNew<RenderedImage>();
    result->image = QImage(size.width(), size.height(), QImage::Format_RGB888);
    result->RK_inverse = program->getInverseRK(sample, size.width(), size.height());

    // Nakopcime data
    int h=size.height();
    for (int y=0; y<h; y++) {
        uchar *dst = result->image.bits() + y*(size.width() * 3);
        uchar *src = pixels + (h-1-y)*(size.width() * 3);

        // copy line
        memcpy(dst, src, size.width() * 3);
    }

    context->doneCurrent();

    return result;
}











}
