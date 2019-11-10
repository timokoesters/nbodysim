#version 450

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

layout(location = 0) out vec3 fragColor;

const float G = 6.67408E-11;
const float MIN_DISTANCE2 = pow(1E8, 2);
const float LIGHT_SPEED = 3E8;
const float LIGHT_SPEED2 = pow(LIGHT_SPEED, 2);

struct Particle {
    vec3 pos; // 0, 1, 2
    vec3 vel; // 4, 5, 6
    float mass; // 7
};

layout(set = 0, binding = 0) uniform GlobalsBuffer {
    mat4 matrix;
    vec3 camera_pos;
    uint particles;
    float delta;
};

layout(std430, set = 0, binding = 1) buffer DataOld {
    Particle data_old[];
};

layout(std430, set = 0, binding = 2) buffer DataCurrent {
    Particle data[];
};

float rand(vec3 co) {
    return (fract(sin(dot(co.xyz, vec3(32.3485, 11.8743, 50.463))) * 48510.7134) - 0.5) * 2.0;
}

float length2(vec3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

void main() {
    int i = gl_VertexIndex;

    // Early return
    if(data_old[i].mass < 0) { 
        gl_PointSize = 0;
        return;
    }

    // Gravity
    if(delta > 0.0) {
        vec3 temp = vec3(0.0, 0.0, 0.0);
        for(int j = 0; j < particles; j++) {
            if(j == i) { continue; }
            if(data_old[j].mass == 0) { break; }

            vec3 diff = data_old[i].pos - data_old[j].pos;

            float d2 = length2(diff);
            if(d2 < MIN_DISTANCE2) { continue; }

            vec3 dir = normalize(diff);
            temp += dir * data_old[j].mass / d2;
        }

        data[i].vel -= temp * G * delta;
        if(length2(data[i].vel) > LIGHT_SPEED2) {
            data[i].vel = normalize(data[i].vel) * LIGHT_SPEED;
        }

        // Update pos
        data[i].pos += data[i].vel * delta;
    }

    // Render
    gl_Position = matrix * vec4(data[i].pos - camera_pos, 1.0);

    gl_PointSize = clamp(data_old[i].mass * 4E-29, 1, 10);

    float red = clamp((length2(data[i].vel)) * 1E-12, 0.0, 1.0);
    fragColor = vec3(red, 0.0, 1-red);
}
