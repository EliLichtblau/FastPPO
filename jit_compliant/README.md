JIT_ENV:
    RotatorWorldJit:
        50k steps:
                not jit -> 9.5 seconds
                jitted -> 3.8 seoncds
        50k observations (stupid implementation):
                not jit -> 2.95
                jitted -> 1.87
        50k rewards (prolly useful to make this faster):
                not jit -> 6.3 seconds
                jitted -> 4.27 seconds
    
    RotatorWorldEnvironment:
        TODO:
            Test if step is actually jitted



JIT_ACTOR_CRITIC:
        ActorCritic:
                Forward:
                        20k steps:
                                not jit -> 8.5 seconds
                                jit -> 5.1 seconds
                evaluate_actions:
                        20k steps:
                                not jit -> 8.2 seconds
                                jit -> 5 seconds
                                