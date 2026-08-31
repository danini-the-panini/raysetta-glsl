#version 410

in vec3 nor;
layout (location = 0) out vec4 outColor;

void main() {
  outColor = vec4(0.5*(nor + vec3(1.0)), 1.0);
}
