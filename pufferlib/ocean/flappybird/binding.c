/*
 * Python bindings for Flappy Bird environment
 */

#include "flappybird.h"
#define Env FlappyBird
#define MY_GET  // Enable custom env_get function
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // Get seed from kwargs
    int seed = (int)unpack(kwargs, "seed");
    env->rand_state = (uint32_t)(seed == 0 ? 1 : seed);
    
    init(env);
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "pipes_passed", log->pipes_passed);
    return 0;
}

// Expose exact game state for replay recording
static PyObject* my_get(PyObject* dict, Env* env) {
    assign_to_dict(dict, "bird_y", env->bird_y);
    assign_to_dict(dict, "bird_velocity", env->bird_velocity);
    assign_to_dict(dict, "score", (float)env->score);
    assign_to_dict(dict, "tick", (float)env->tick);
    
    // Create pipes list with exact values
    PyObject* pipes = PyList_New(NUM_PIPES);
    for (int i = 0; i < NUM_PIPES; i++) {
        PyObject* pipe = PyList_New(3);
        PyList_SetItem(pipe, 0, PyFloat_FromDouble(env->pipes[i].x));
        PyList_SetItem(pipe, 1, PyFloat_FromDouble(env->pipes[i].gap_y));
        PyList_SetItem(pipe, 2, PyFloat_FromDouble(env->pipes[i].passed));
        PyList_SetItem(pipes, i, pipe);
    }
    PyDict_SetItemString(dict, "pipes", pipes);
    Py_DECREF(pipes);
    
    return dict;
}
