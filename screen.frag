#version 410
precision highp float;

#define sigmaS 2.0
#define sigmaL 0.1

#define EPS 1e-5

uniform sampler2D frame;
uniform sampler2D raster;
uniform sampler2D raster_depth;

in vec2 uv;
layout (location = 0) out vec4 outColor;

vec4 texdepth(sampler2D tex, sampler2D depth, vec2 uv) {
  return vec4(texture(tex, uv).rgb, texture(depth, uv).r);
}

vec4 bilat(sampler2D image, sampler2D reg, sampler2D regd, vec2 uv) {
  float sigS = max(sigmaS, EPS);
  float sigL = max(sigmaL, EPS);

  float facS = -1.0/(2.0*sigS*sigS);
  float facL = -1.0/(2.0*sigL*sigL);

  float sumW = 0.0;
  vec4  sumC = vec4(0.0);
  float halfSize = sigS * 2.0;
  ivec2 textureSize2 = textureSize(image, 0);

  vec4 l = texture(reg,uv);

  for (float i = -halfSize; i <= halfSize; i ++){
    for (float j = -halfSize; j <= halfSize; j ++){
      vec2 pos = vec2(i, j);
      vec2 uv2 = uv + pos / vec2(textureSize2);
      vec4 offsetColor = texture(image, uv2);
      vec4 offsetReg = texdepth(reg, regd, uv2);

      float distS = length(pos);
      float distL = length(offsetReg.rgb-l.rgb);

      float wS = exp(facS*float(distS*distS));
      float wL = exp(facL*float(distL*distL));
      float w = wS*wL;

      sumW += w;
      sumC += offsetColor * w;
    }
  }

  return sumC/sumW;
}

void main() {
  vec2 st = vec2(uv.x, 1.0 - uv.y);
  float depth = (1.0-texture(raster_depth, st).r)*1000.0;
  // outColor = vec4(texture(raster, st).rgb * vec3(depth*1000.0), 1.0);
  // outColor = mix(
  //   ,
  //   texture(raster, st),
  //   0.5
  // );

  vec3 rt = pow(clamp(bilat(frame, raster, raster_depth, st).rgb, 0.0, 1.0), vec3(1.0/2.2));
  vec3 rs = texture(raster, st).rgb;
  // outColor = vec4(mix(rt, rs, 0.5), 1.0);
  outColor = vec4(rt, 1.0);
}
