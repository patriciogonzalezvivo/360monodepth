#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D   u_tex0;
uniform sampler2D   u_tex1;

uniform vec2        u_resolution;
uniform float       u_time;

varying vec4        v_position;
varying vec4        v_color;
varying vec3        v_normal;

#ifdef MODEL_VERTEX_TEXCOORD
varying vec2        v_texcoord;
#endif

#ifdef MODEL_VERTEX_TANGENT
varying vec4        v_tangent;
varying mat3        v_tangentToWorld;
#endif

void main(void) {
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    vec2 pixel = 1.0/u_resolution;
    vec2 st = gl_FragCoord.xy * pixel;
    vec2 uv = v_texcoord;
    
    color = texture2D(u_tex0, uv);

    gl_FragColor = color;
}
