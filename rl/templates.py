# Episode/probe prompt templates for the grader-RL line.
#
# V1 = the Arm-1 phrasing (original task prompt + "Here are two examples of
# student responses..."), used by the 2026-08-29 signal run and all probes to
# date. V2 = Sanjeev's 2026-08-30 restructuring with two fixes agreed in chat:
# "attempts" not "solutions" (the task's own definition makes "solution" imply
# validity, which x- lacks), and an instruction that closes the
# faithful-but-wrong loophole ("complete their trace" would license continuing
# a wrong prefix to its wrong conclusion; the objective is reach-a-correct-
# answer, fixing errors). The trace framing is first-person because the prefix
# occupies the model's own <think> channel in Qwen3's format -- a third-person
# "trace below" would describe content the model encounters as its own
# thought.
#
# COUPLING: whatever template trains the grader must also be the OPSD teacher
# context and the probe context in that run's evals. Template is a design
# parameter that moves everywhere together.

V2_HEAD = (
    "In the COUNTDOWN task you are given a LIST of integers and a TARGET "
    "integer. A solution is an arithmetic expression that uses basic "
    "arithmetic operations (+, -, *, /) and each number in LIST exactly "
    "once, and evaluates to the TARGET.\n\n"
    "For example, if LIST was [2, 3, 5] and TARGET was 10, then "
    "\\boxed{{2+3+5}} could be an answer.\n\n"
    "Now consider the following instance:\n\n"
    "LIST: [{nums}]\n"
    "TARGET: {target}\n"
)

# Episode variant: a partial reasoning trace will occupy the <think> channel.
V2_PAIR_EPISODE = (
    "\nBelow are two student attempts at this instance.\n\n"
    "ATTEMPT A:\n{a}\n\n"
    "ATTEMPT B:\n{b}\n\n"
    "Now work out your own solution. Continue your reasoning from where it "
    "leaves off, fixing any errors along the way, and finish with a correct "
    "final answer within \\boxed{{}}."
)

# V3 (Sanjeev 2026-08-30): discourage explicit references to the attempts so
# pair-assisted traces read as organic from-scratch solving (usable as
# bare-question training data). His phrasing + "or to the students" (38.8% of
# v2 traces said "student" without "attempt").
V3_PAIR_PROBE = (
    "\nBelow, for your private reference only, are two student attempts at "
    "this instance.\n\n"
    "ATTEMPT A:\n{a}\n\n"
    "ATTEMPT B:\n{b}\n\n"
    "Your reasoning can make use of them but do not explicitly mention, "
    "quote, or refer to the attempts or to the students in either your "
    "reasoning or while writing your answer. A reader of your reasoning "
    "should not be able to tell you saw them.\n\n"
    "Now work out your own solution. Reason step by step and finish with a "
    "correct final answer within \\boxed{{}}."
)

# Probe variant: no prefix, so no continue clause.
V2_PAIR_PROBE = (
    "\nBelow are two student attempts at this instance.\n\n"
    "ATTEMPT A:\n{a}\n\n"
    "ATTEMPT B:\n{b}\n\n"
    "Now work out your own solution. Reason step by step and finish with a "
    "correct final answer within \\boxed{{}}."
)

# NOTE: not run through .format() -- single braces.
V2_BARE = (
    "\nWork out a solution. Reason step by step and put your final answer "
    "within \\boxed{}."
)


def v2_pair_content(nums, target, a, b, episode=False):
    tail = V2_PAIR_EPISODE if episode else V2_PAIR_PROBE
    return (V2_HEAD.format(nums=", ".join(str(n) for n in nums),
                           target=target)
            + tail.format(a=a, b=b))


def v3_pair_content(nums, target, a, b):
    return (V2_HEAD.format(nums=", ".join(str(n) for n in nums),
                           target=target)
            + V3_PAIR_PROBE.format(a=a, b=b))


def v2_bare_content(nums, target):
    return (V2_HEAD.format(nums=", ".join(str(n) for n in nums),
                           target=target)
            + V2_BARE)
