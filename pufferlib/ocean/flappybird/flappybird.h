/*
 * Flappy Bird Environment for PufferLib Ocean
 * 
 * A minimal, headless, deterministic Flappy Bird implementation with:
 * - Discrete actions: 0 = no-op, 1 = flap
 * - Vector observations (no pixels)
 * - Simple physics (gravity/flap)
 * - Pipes, collisions, score tracking
 */

#ifndef FLAPPYBIRD_H
#define FLAPPYBIRD_H

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>

// Game constants
#define SCREEN_HEIGHT 512.0f
#define SCREEN_WIDTH 288.0f
#define BIRD_X 50.0f

// Physics
#define GRAVITY 0.5f
#define FLAP_VELOCITY -8.0f
#define MAX_VELOCITY 10.0f

// Bird dimensions
#define BIRD_RADIUS 12.0f

// Pipe settings
#define PIPE_WIDTH 52.0f
#define PIPE_GAP 100.0f
#define PIPE_VELOCITY -4.0f
#define PIPE_SPAWN_X (SCREEN_WIDTH + PIPE_WIDTH)
#define MIN_PIPE_HEIGHT 50.0f

// Limits
#define MAX_STEPS 2000
#define NUM_PIPES 2
#define NUM_OBS 8

// Logging structure - must only contain floats
typedef struct Log {
    float score;
    float episode_length;
    float episode_return;
    float pipes_passed;
    float n;
} Log;

// Pipe structure
typedef struct Pipe {
    float x;
    float gap_y;  // Center of the gap
    int passed;
} Pipe;

// Main environment structure
typedef struct FlappyBird {
    // Buffers (set by Python)
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    
    // Logging
    Log log;
    
    // Bird state
    float bird_y;
    float bird_velocity;
    
    // Pipe state
    Pipe pipes[NUM_PIPES];
    
    // Game state
    int score;
    int tick;
    float episode_return;
    
    // Random state
    uint32_t rand_state;
    
    // No client for headless
    void* client;
} FlappyBird;

// Fast pseudo-random number generator (xorshift32)
static inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Random float in range [min, max)
static inline float rand_float(uint32_t* state, float min, float max) {
    return min + (xorshift32(state) / (float)UINT32_MAX) * (max - min);
}

// Spawn a new pipe
static inline void spawn_pipe(FlappyBird* env, int pipe_idx, float base_x) {
    float min_gap = MIN_PIPE_HEIGHT + PIPE_GAP / 2;
    float max_gap = SCREEN_HEIGHT - MIN_PIPE_HEIGHT - PIPE_GAP / 2;
    
    env->pipes[pipe_idx].x = base_x;
    env->pipes[pipe_idx].gap_y = rand_float(&env->rand_state, min_gap, max_gap);
    env->pipes[pipe_idx].passed = 0;
}

// Compute normalized observations
static void compute_observations(FlappyBird* env) {
    // Bird state (normalized to roughly [-1, 1])
    env->observations[0] = (env->bird_y - SCREEN_HEIGHT / 2) / (SCREEN_HEIGHT / 2);
    env->observations[1] = env->bird_velocity / MAX_VELOCITY;
    
    // Find pipes in order of x position
    int first = (env->pipes[0].x <= env->pipes[1].x) ? 0 : 1;
    int second = 1 - first;
    
    // Check if first pipe has passed bird
    if (env->pipes[first].x + PIPE_WIDTH / 2 < BIRD_X - BIRD_RADIUS) {
        // Swap to get the next pipe
        int tmp = first;
        first = second;
        second = tmp;
    }
    
    // First pipe info
    float pipe_x = env->pipes[first].x;
    float gap_y = env->pipes[first].gap_y;
    env->observations[2] = (pipe_x - BIRD_X) / SCREEN_WIDTH;
    env->observations[3] = (gap_y - PIPE_GAP / 2 - SCREEN_HEIGHT / 2) / (SCREEN_HEIGHT / 2);
    env->observations[4] = (gap_y + PIPE_GAP / 2 - SCREEN_HEIGHT / 2) / (SCREEN_HEIGHT / 2);
    
    // Second pipe info
    pipe_x = env->pipes[second].x;
    gap_y = env->pipes[second].gap_y;
    env->observations[5] = (pipe_x - BIRD_X) / SCREEN_WIDTH;
    env->observations[6] = (gap_y - PIPE_GAP / 2 - SCREEN_HEIGHT / 2) / (SCREEN_HEIGHT / 2);
    env->observations[7] = (gap_y + PIPE_GAP / 2 - SCREEN_HEIGHT / 2) / (SCREEN_HEIGHT / 2);
}

// Check for collisions
static bool check_collision(FlappyBird* env) {
    float bird_y = env->bird_y;
    
    // Ceiling/floor collision
    if (bird_y - BIRD_RADIUS < 0 || bird_y + BIRD_RADIUS > SCREEN_HEIGHT) {
        return true;
    }
    
    // Pipe collisions
    for (int i = 0; i < NUM_PIPES; i++) {
        float pipe_x = env->pipes[i].x;
        float gap_y = env->pipes[i].gap_y;
        
        // Check if bird is horizontally within pipe
        if (BIRD_X + BIRD_RADIUS > pipe_x - PIPE_WIDTH / 2 &&
            BIRD_X - BIRD_RADIUS < pipe_x + PIPE_WIDTH / 2) {
            
            // Check if bird is outside the gap
            float top_y = gap_y - PIPE_GAP / 2;
            float bottom_y = gap_y + PIPE_GAP / 2;
            
            if (bird_y - BIRD_RADIUS < top_y || bird_y + BIRD_RADIUS > bottom_y) {
                return true;
            }
        }
    }
    
    return false;
}

// Add to log on episode end
static void add_log(FlappyBird* env) {
    env->log.score += env->score;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.pipes_passed += env->score;
    env->log.n += 1;
}

// Initialize environment (called once per env)
static void init(FlappyBird* env) {
    env->tick = 0;
    memset(&env->log, 0, sizeof(Log));
}

// Reset environment
static void c_reset(FlappyBird* env) {
    env->bird_y = SCREEN_HEIGHT / 2;
    env->bird_velocity = 0.0f;
    env->score = 0;
    env->tick = 0;
    env->episode_return = 0.0f;
    
    // Spawn initial pipes
    spawn_pipe(env, 0, PIPE_SPAWN_X);
    spawn_pipe(env, 1, PIPE_SPAWN_X + 200);
    
    compute_observations(env);
}

// Step environment
static void c_step(FlappyBird* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.1f;  // Alive bonus
    
    // Apply flap if action is 1
    float action = env->actions[0];
    if (action > 0.5f) {
        env->bird_velocity = FLAP_VELOCITY;
    }
    
    // Apply gravity
    env->bird_velocity += GRAVITY;
    
    // Clamp velocity
    if (env->bird_velocity > MAX_VELOCITY) env->bird_velocity = MAX_VELOCITY;
    if (env->bird_velocity < -MAX_VELOCITY) env->bird_velocity = -MAX_VELOCITY;
    
    // Update position
    env->bird_y += env->bird_velocity;
    
    // Move pipes and check for scoring
    for (int i = 0; i < NUM_PIPES; i++) {
        env->pipes[i].x += PIPE_VELOCITY;
        
        // Check if bird passed this pipe
        float pipe_x = env->pipes[i].x;
        if (!env->pipes[i].passed && pipe_x + PIPE_WIDTH / 2 < BIRD_X) {
            env->pipes[i].passed = 1;
            env->score += 1;
            env->rewards[0] = 1.0f;  // Pipe pass bonus
        }
        
        // Respawn pipe if off screen
        if (pipe_x < -PIPE_WIDTH) {
            spawn_pipe(env, i, PIPE_SPAWN_X);
        }
    }
    
    env->tick += 1;
    
    // Check for collision or timeout
    bool collision = check_collision(env);
    bool timeout = env->tick >= MAX_STEPS;
    
    if (collision) {
        env->rewards[0] = -1.0f;  // Death penalty
        env->terminals[0] = 1;
    }
    
    env->episode_return += env->rewards[0];
    
    // Handle episode end
    if (collision || timeout) {
        add_log(env);
        c_reset(env);
    }
    
    compute_observations(env);
}

// Render (headless - no-op)
static void c_render(FlappyBird* env) {
    (void)env;
}

// Close (no resources to free)
static void c_close(FlappyBird* env) {
    (void)env;
}

#endif // FLAPPYBIRD_H
