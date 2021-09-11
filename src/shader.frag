#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    //outColor = vec4(fragColor, 1.0);

    dvec3 diff = dvec3(0, 0, 1);
    if (isnan((diff / 1.0).x)) { outColor = vec4(1.0, 0.0, 0.0, 1.0); }
    else { outColor = vec4(0.0, 1.0, 0.0, 1.0); }
}
