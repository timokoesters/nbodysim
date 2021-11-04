#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    //outColor = vec4(fragColor, 1.0);

    dvec3 test = dvec3(1.0, 0.0, 0.0);
    if((test / 1.0).x - 1.0 > 1000000) { outColor = vec4(1.0, 0.0, 0.0, 1.0); }
    else { outColor = vec4(0.0, 1.0, 0.0, 1.0); }
}
