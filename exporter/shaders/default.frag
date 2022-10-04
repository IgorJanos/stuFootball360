
uniform sampler2D texture;
uniform sampler1D textureDistort;
varying highp vec2 t;

uniform mat3 rk;
uniform highp vec2 canvas;
uniform vec4 args;
uniform vec4 k;

const float PI = 3.1415926535897932384626433832795;
const float PI_2 = 1.57079632679489661923;


float distortRate(float r, float rMax)
{
    vec4 s = texture1D(textureDistort, r / rMax);
    return s.r;
}


vec2 distort(vec2 p, vec2 c, float rMax)
{
    p = p - c;
    float r = length(p);
    float d = distortRate(r, rMax);
    return c + (p * d);
}

vec3 gamma(vec3 c, float gamma)
{
    return pow(c, vec3(1. / gamma));
}

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}


vec2 reproject(vec2 p)
{
    vec2    result;
    vec3    dp = normalize(vec3(p.x, p.y, 1.0));
    vec3    r = rk * dp;

    float   theta, phi;
    theta = atan(r.x, r.z);
    phi   = atan(r.y, sqrt(r.x*r.x + r.z*r.z));

    // result
    result.x = (((theta * 1.0) / PI + 1.0) / 2.0);
    result.y = (phi + PI_2) * 1.0 / PI;
    return result;
}

vec4 process(vec4 color)
{
    vec3 c = color.rgb;

    // HSV
    vec3 _hsv = rgb2hsv(c);
    _hsv.x += args.x;
    _hsv.yz *= args.yz;
    _hsv.x = mod(_hsv.x, 1.0);
    _hsv.yz = clamp(_hsv.yz, 0.0, 1.0);
    c = hsv2rgb(_hsv);

    // gamma correction
    c = gamma(c, args[3]);

    return vec4(c.rgb, 1.0);
}


void main()
{
    float w = 0.5*canvas.x;
    float rMax = sqrt(0.5*0.5 + w*w);

    // Rescale for aspect ratio
    vec2 i = t * canvas;
    vec2 c = vec2(0.5, 0.5) * canvas;

    // compute distortion & reprojection
    vec2 distortedPos = distort(i, c, rMax);
    vec2 rep = reproject(distortedPos);
    vec4 color = texture2D(texture, rep);

    // Result color
    gl_FragColor = process(color);
}
