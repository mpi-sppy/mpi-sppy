.. _Spokes:

Spokes
======

In this section we provide an overview of some of the spoke classes
that are part of the ``mpi-sppy`` release.


Outer Bound
-----------

For minimization problems, `outer bound` means `lower bound`.

Frank-Wolf
^^^^^^^^^^^

This bound is based on the paper `Combining Progressive Hedging with a Frank--Wolfe Method to Compute Lagrangian Dual Bounds in Stochastic Mixed-Integer Programming` by Boland et al. It does not receive information from the hub, it
simply sends bounds as they are available. Compared to the Lagrangian bounds,
it takes longer to compute but is generally tighter once it reports a bound.


Lagrangian
^^^^^^^^^^

This bound is based on the paper `Obtaining lower bounds from the progressive hedging algorithm for stochastic mixed-integer programs` by Gade et al. It takes
W values from the hub and uses them to compute a bound.


Lagranger
^^^^^^^^^

This bound is a variant of the Lagrangian bound, but it takes x values from the
hub and uses those to compute its own W values. It can modify the rho
values (typically to use lower values). The modification is done
in the form of scaling factors that are specified to be applied at a given
iteration. The factors accumulate so if 0.5 is applied at iteration 1 and
1.2 is applied at iteration 5, from iteration 5 onward, the factor will be 1.2.

Inner Bounds
------------

For minimization problems, `inner bound` means `upper bound`. But more
importantly, the bounds are based on a solution, whose value can be
computed. In some sense, if you don't have this solution, you don't
have anything (even if you think your hub algorithm has `converged` in
some sense). We refer to this solution as xhat (:math:`\hat{x}`)

xhat_specific_bounder
^^^^^^^^^^^^^^^^^^^^^

At construction, this spoke takes a specification of a scenario per
non-leaf node of the scenario tree (so for a two-stage problem, one
scenario), which are used at every iteration of the hub algorithm as
trial values for :math:`\hat{x}`.

xhatshufflelooper_bounder
^^^^^^^^^^^^^^^^^^^^^^^^^

This bounder shuffles the scenarios loops over them scenarios until
the hub provides a new x.  To ensure that all subproblems are tried
eventually, the spoke remembers where it left off, and resumes from
its prior position.  Since the resulting subproblems after fixing the
first-stage variables are usually much easier to solve, many candidate
solutions can be tried before receiving new x values from the hub.

slam_heuristic
^^^^^^^^^^^^^^

This heuristic attempts to find a feasible solution by slamming every
variable to its maximum (or minimum) over every scenario associated 
with that scenario tree node.


General
-------

cross scenario
^^^^^^^^^^^^^^

Passes cross scenario cuts.


