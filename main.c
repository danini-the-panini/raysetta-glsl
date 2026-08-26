#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#define GLAD_GL_IMPLEMENTATION
#include "gl.h"

#include <GLFW/glfw3.h>

#define WIDTH 400
#define HEIGHT 200

typedef struct sphere {
  struct { float x; float y; float z; } center;
  float radius;
} Sphere;

static inline void sph_set_center(Sphere *sph, float x, float y, float z) {
  sph->center.x = x;
  sph->center.y = y;
  sph->center.z = z;
}

static bool load_shader(const char *filename, GLenum type, GLuint *ret) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "could not open file %s\n", filename);
    return false;
  }

  fseek(file, 0, SEEK_END);
  size_t length = ftell(file);
  fseek(file, 0, SEEK_SET);
  GLchar* str = malloc(length+1);
  fread(str, 1, length, file);
  str[length] = '\0';

  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &str, NULL);
  glCompileShader(shader);
  free(str);
  *ret = shader;
  return true;
}

void print_shader_info_log(GLuint shader) {
  GLint len = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
  if (len <= 0) return;

  GLchar *log = malloc(len);
  glGetShaderInfoLog(shader, len, &len, log);

  fprintf(stderr, "%s\n", log);
}

void print_program_info_log(GLuint program) {
  GLint len = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
  if (len <= 0) return;

  GLchar *log = malloc(len);
  glGetProgramInfoLog(program, len, &len, log);

  fprintf(stderr, "%s\n", log);
  free(log);
}

bool check_program(GLuint program) {
  GLint link_status = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &link_status);
  if (link_status == GL_TRUE) return true;
  return false;
}

int main(void) {
  srand(time(NULL));
  glfwInit();

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "raysetta-gl", NULL, NULL);
  glfwMakeContextCurrent(window);

  int version = gladLoadGL(glfwGetProcAddress);
  printf("GL %d.%d\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));

  GLsizei max_tex = 0;
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_tex);
  printf("MAX TEX: %i\n", max_tex);

  float *noise = malloc(sizeof(float)*max_tex);
  for (GLsizei i = 0; i < max_tex; i++) {
    noise[i] = (float)rand() / (float)RAND_MAX;
  }

  GLuint vao;
  glGenVertexArrays(1, &vao);

  GLuint vert, frag;
  if (!load_shader("full_screen.vert", GL_VERTEX_SHADER, &vert)) return 1;
  if (!load_shader("rt.frag", GL_FRAGMENT_SHADER, &frag)) return 1;

  const GLuint program = glCreateProgram();
  glAttachShader(program, vert);
  glAttachShader(program, frag);
  glLinkProgram(program);

  if (!check_program(program)) {
    print_shader_info_log(vert);
    print_shader_info_log(frag);
    print_program_info_log(program);
    return 1;
  }

  glUseProgram(program);

  GLuint noise_tex = 0;
  glGenTextures(1, &noise_tex);
  glBindTexture(GL_TEXTURE_1D, noise_tex);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexStorage1D(GL_TEXTURE_1D, 1, GL_R32F, max_tex);
  glTexSubImage1D(GL_TEXTURE_1D, 0, 0, max_tex, GL_RED, GL_FLOAT, noise);
  glActiveTexture(GL_TEXTURE0);

  GLuint noise_loc = glGetUniformLocation(program, "noise");
  GLuint noise_size_loc = glGetUniformLocation(program, "noise_size");
  glUniform1i(noise_loc, 0);
  glUniform1i(noise_size_loc, max_tex);

  const int samples = 5;

  GLuint samples_loc = glGetUniformLocation(program, "samples");
  glUniform1i(samples_loc, samples);

  GLuint res_loc = glGetUniformLocation(program, "res");
  glUniform2f(res_loc, WIDTH, HEIGHT);

  GLuint cam_eye_loc = glGetUniformLocation(program, "cam_eye");
  glUniform3f(cam_eye_loc, 0.0, 0.0, 0.0);

  const int sphere_count = 2;
  Sphere spheres[100];

  sph_set_center(&spheres[0], 0.0, 0.0, -1.0);
  spheres[0].radius = -0.5;

  sph_set_center(&spheres[1], 0.0, -100.5, -1.0);
  spheres[1].radius = 100.0;

  GLuint sphere_count_loc = glGetUniformLocation(program, "sphere_count");
  glUniform1i(sphere_count_loc, sphere_count);

  GLuint sph_buf;
  glGenBuffers(1, &sph_buf);
  glBindBuffer(GL_UNIFORM_BUFFER, sph_buf);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(Sphere)*100, NULL, GL_STATIC_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  GLuint sph_loc = glGetUniformBlockIndex(program, "ObjectBloc");
  glUniformBlockBinding(program, sph_loc, 0);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, sph_buf);
  glBindBufferRange(GL_UNIFORM_BUFFER, 2, sph_buf, 0, sizeof(Sphere)*100);

  glBindBuffer(GL_UNIFORM_BUFFER, sph_buf);
  glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Sphere)*sphere_count, spheres);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    int w, h;
    glfwGetWindowSize(window,&w, &h);
    glUniform2f(res_loc, w, h);

    for (GLsizei i = 0; i < max_tex; i++) {
      noise[i] = (float)rand() / (float)RAND_MAX;
    }
    glTexSubImage1D(GL_TEXTURE_1D, 0, 0, max_tex, GL_RED, GL_FLOAT, noise);

    glClearColor(0.7f, 0.9f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(window);
  }

  glfwTerminate();

  free(noise);
  return 0;
}
