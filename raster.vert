#version 410

uniform mat4 view_proj;

in vec3 position;
in vec3 normal;

out vec3 nor;

void main() {
  gl_Position = view_proj * vec4(position, 1.0);
  nor = normal;
}
