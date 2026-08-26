#version 330

out vec2 uv;
void main()
{
  vec3 vertices[3] = vec3[3](
    vec3(-3.0, -1.0, 0.0),
    vec3(3.0, -1.0, 0.0),
    vec3(0.0, 2.0, 0.0)
  );

  vec3 position = vertices[gl_VertexID];

  uv = vec2(0.5 * position.x + 0.5, 0.5 - 0.5 * position.y);
  gl_Position = vec4(position, 1.0);
}
