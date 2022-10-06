//-----------------------------------------------------------------------------
//
//  Football360 Exporter
//
//  Author : Igor Janos
//
//-----------------------------------------------------------------------------
#ifndef EXPORTER_H
#define EXPORTER_H



#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include <QOpenGLTexture>


namespace Exporter {

//-----------------------------------------------------------------------------
//
//  Exporting API
//
//-----------------------------------------------------------------------------

template<class T>
class PipelineSource
{
public:

    virtual QSharedPointer<T> current() = 0;
    virtual void reset() = 0;
    virtual void next() = 0;
    virtual bool hasCurrent() = 0;

};

class Image
{
public:
    QString         filename;       // 001.jpg
    QImage          image;
};

class RenderedImage
{
public:
    QImage          image;
    cv::Mat         RK_inverse;
};

class DatasetImageSource : public PipelineSource<Image>
{
protected:

    QMutex              lock;

    QString             path;
    QStringList         images;
    int                 index;

public:
    DatasetImageSource(QString apath, QStringList aimages);

    // PipelineSource
    virtual QSharedPointer<Image> current();
    virtual void reset();
    virtual void next();
    virtual bool hasCurrent();

};

class CycleCounter : public PipelineSource<Image>
{
protected:

    QMutex              lock;

    QSharedPointer<PipelineSource<Image>>       source;
    int                                         count;
    int                                         index;

public:
    CycleCounter(QSharedPointer<PipelineSource<Image>> asrc, int acount);

    // PipelineSource
    virtual QSharedPointer<Image> current();
    virtual void reset();
    virtual void next();
    virtual bool hasCurrent();

};

class Repeater : public PipelineSource<Image>
{
protected:

    QMutex              lock;

    QSharedPointer<Image>                       cache;
    QSharedPointer<PipelineSource<Image>>       source;
    int                                         count;
    int                                         index;

public:
    Repeater(QSharedPointer<PipelineSource<Image>> asrc, int acount);

    // PipelineSource
    virtual QSharedPointer<Image> current();
    virtual void reset();
    virtual void next();
    virtual bool hasCurrent();

};

class CropSample
{
public:

    float           p, t, r;
    float           fov;
    float           k1, k2;

public:
    CropSample();
    CropSample(const CropSample &v);
    CropSample &operator =(CropSample v);

};

class InterpolatedFunction
{
protected:

    bool                                    invertible;
    std::vector<std::pair<float,float>>     data;
    float                                   xMin, xMax, yMin, yMax;

public:
    InterpolatedFunction();

    void add(float x, float y);
    void reset();

    float getY(float x);
    float getX(float y);

    std::pair<float,float> xRange();
    std::pair<float,float> yRange();
};




class DatasetSink
{
protected:

    int                             totalCount;
    int                             writtenCount;
    QSize                           scaleSize;

    QSharedPointer<H5::H5File>      file;
    QSharedPointer<H5::Group>       groupImages;


public:
    DatasetSink(
            const char *afilename,
            QSize ascalesize,
            int atotalcount
            );
    virtual ~DatasetSink();

    // Write
    void write(QSharedPointer<RenderedImage> frame, CropSample sample);
    bool isComplete();

};



class PinholeProgram
{
private:

    QOpenGLFunctions        *f;
    bool                    heightWise;

    // Rendering program
    QOpenGLShaderProgram    *program;
    GLint                   posVertex;
    GLint                   posTex;
    GLint                   posTexture;
    GLint                   posDistortTexture;
    GLint                   posCanvas;
    GLint                   posRK;
    GLint                   posArgs;
    GLint                   posK;

    void setCanvas(QVector2D value);
    void setRK(QMatrix3x3 value);

public:
    PinholeProgram(QOpenGLFunctions *func);

    void init();
    void destroy();
    void bind();
    void unbind();

    void prepareView(CropSample s, int width, int height);
    cv::Mat getInverseRK(CropSample s, int width, int height);

    void setTexture(GLint value);
    void setDistortTexture(GLint value);
    void setArgs(float gamma, QVector3D hsv);
    void setK(QVector4D k);

    void draw();

    inline bool HeightWise() { return heightWise; }
};



class CropRenderer
{
protected:

    QSize                   size;
    QOpenGLContext          *context;
    QOffscreenSurface       *surface;
    PinholeProgram          *program;

    GLuint                  rtTarget, dsTarget;
    GLuint                  fbo;
    uchar                   *pixels;

    QOpenGLTexture          *texture;
    QOpenGLTexture          *texDistort;
    QSharedPointer<Image>   lastImage;

    InterpolatedFunction    dist;


    bool isInitialized;

protected:

    bool initialize();

    void computeDistortTexture(InterpolatedFunction &func, float maxR, bool inverse);

public:
    CropRenderer(QOffscreenSurface *asurface, QSize asize);
    virtual ~CropRenderer();

    // Rendering
    QSharedPointer<RenderedImage> render(QSharedPointer<Image> image, CropSample sample);

};



};


#endif // EXPORTER_H
