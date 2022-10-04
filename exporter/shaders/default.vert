

attribute highp vec4 vertex;
attribute highp vec2 tex;

varying highp vec2 t;

void main()
{
    t = tex;
    gl_Position = vertex;
}
