#version 410
precision highp float;

uniform sampler2D frame;

in vec2 uv;
layout (location = 0) out vec4 outColor;

void main() {
  outColor = vec4(pow(clamp(texture(frame, vec2(uv.x, 1.0 - uv.y)).rgb, 0.0, 1.0), vec3(1.0/2.2)), 1.0);
}
