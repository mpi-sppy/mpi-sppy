###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Phase-1 tests for solver_options_layers — see
# doc/designs/solver_options_redesign.md §6.4. The contract this file
# pins is: the new layered representation folds (per-iteration) to
# dicts identical to the legacy iter0_solver_options /
# iterk_solver_options dicts produced by shared_options and
# apply_solver_specs. Phase 1 is a pure-refactor groundwork step;
# every later phase will lean on this equivalence.

import copy
import unittest

from mpisppy.utils import config
from mpisppy.utils.cfg_vanilla import shared_options, apply_solver_specs
from mpisppy.utils.sputils import fold_solver_options_layers


def _bare_cfg():
    cfg = config.Config()
    cfg.popular_args()
    cfg.add_mipgap_specs()
    return cfg


def _spoke_cfg(spoke_name):
    cfg = _bare_cfg()
    cfg.add_solver_specs(prefix=spoke_name)
    cfg.add_mipgap_specs(prefix=spoke_name)
    return cfg


class TestSharedOptionsLayers(unittest.TestCase):

    def test_empty_cfg_yields_empty_layers(self):
        cfg = _bare_cfg()
        sh = shared_options(cfg)
        self.assertEqual(sh["solver_options_layers"], [])
        self.assertEqual(
            fold_solver_options_layers(sh["solver_options_layers"], 0), {})
        self.assertEqual(
            fold_solver_options_layers(sh["solver_options_layers"], 1), {})

    def test_solver_options_string_is_default_layer(self):
        cfg = _bare_cfg()
        cfg.solver_options = "mipgap=0.01 logfile=run.log"
        sh = shared_options(cfg)
        layers = sh["solver_options_layers"]
        expected = {"mipgap": 0.01, "logfile": "run.log"}
        self.assertEqual(fold_solver_options_layers(layers, 0), expected)
        self.assertEqual(fold_solver_options_layers(layers, 1), expected)
        self.assertEqual(fold_solver_options_layers(layers, 7), expected)

    def test_iter0_iterk_mipgap_yield_predicate_layers(self):
        cfg = _bare_cfg()
        cfg.iter0_mipgap = 0.01
        cfg.iterk_mipgap = 0.02
        sh = shared_options(cfg)
        layers = sh["solver_options_layers"]
        self.assertEqual(
            fold_solver_options_layers(layers, 0), {"mipgap": 0.01})
        self.assertEqual(
            fold_solver_options_layers(layers, 1), {"mipgap": 0.02})
        self.assertEqual(
            fold_solver_options_layers(layers, 5), {"mipgap": 0.02})

    def test_max_solver_threads_overrides_solver_options_threads(self):
        # Mirrors today's behavior at cfg_vanilla.py:83-85: the global
        # thread cap overwrites whatever 'threads' the user wrote inline.
        cfg = _bare_cfg()
        cfg.solver_options = "mipgap=0.01 threads=2"
        cfg.max_solver_threads = 4
        sh = shared_options(cfg)
        layers = sh["solver_options_layers"]
        folded = fold_solver_options_layers(layers, 0)
        self.assertEqual(folded["mipgap"], 0.01)
        self.assertEqual(folded["threads"], 4)

    def test_combined_fold_equals_legacy_iter0_iterk_dicts(self):
        # The regression contract for phase 1: under any cfg
        # combination, fold(layers, 0) must match iter0_solver_options
        # and fold(layers, k>=1) must match iterk_solver_options.
        cfg = _bare_cfg()
        cfg.solver_options = "logfile=run.log threads=2"
        cfg.max_solver_threads = 4
        cfg.iter0_mipgap = 0.01
        cfg.iterk_mipgap = 0.02
        sh = shared_options(cfg)
        layers = sh["solver_options_layers"]
        self.assertEqual(
            fold_solver_options_layers(layers, 0),
            sh["iter0_solver_options"],
        )
        self.assertEqual(
            fold_solver_options_layers(layers, 1),
            sh["iterk_solver_options"],
        )
        self.assertEqual(
            fold_solver_options_layers(layers, 7),
            sh["iterk_solver_options"],
        )


class TestApplySolverSpecsLayers(unittest.TestCase):

    def _spoke_dict_from(self, sh):
        # apply_solver_specs operates on spoke["opt_kwargs"]["options"];
        # it expects a deepcopy of shared_options output.
        return {"opt_kwargs": {"options": copy.deepcopy(sh)}}

    def test_per_spoke_solver_options_replace_layers(self):
        # Phase 1 mirrors today's replace-not-overlay semantics in
        # apply_solver_specs (cfg_vanilla.py:119-120). Phase 5 will
        # change this to overlay; phase-1 tests pin the legacy contract.
        cfg = _spoke_cfg("lagrangian")
        cfg.solver_options = "logfile=run.log"
        cfg.lagrangian_solver_options = "mipgap=0.001"
        sh = shared_options(cfg)
        spoke = self._spoke_dict_from(sh)
        apply_solver_specs("lagrangian", spoke, cfg)
        opts = spoke["opt_kwargs"]["options"]
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 0),
            opts["iter0_solver_options"],
        )
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 1),
            opts["iterk_solver_options"],
        )
        self.assertEqual(opts["iter0_solver_options"], {"mipgap": 0.001})

    def test_per_spoke_iter0_iterk_mipgap_layer_predicate(self):
        cfg = _spoke_cfg("lagrangian")
        cfg.solver_options = "logfile=run.log"
        cfg.lagrangian_iter0_mipgap = 0.01
        cfg.lagrangian_iterk_mipgap = 0.02
        sh = shared_options(cfg)
        spoke = self._spoke_dict_from(sh)
        apply_solver_specs("lagrangian", spoke, cfg)
        opts = spoke["opt_kwargs"]["options"]
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 0),
            opts["iter0_solver_options"],
        )
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 1),
            opts["iterk_solver_options"],
        )

    def test_per_spoke_threads_reapplied(self):
        # apply_solver_specs re-applies max_solver_threads at the end
        # (cfg_vanilla.py:127-129). The layered version must produce
        # the same final folded dict for both predicates.
        cfg = _spoke_cfg("lagrangian")
        cfg.solver_options = "mipgap=0.01"
        cfg.lagrangian_solver_options = "presolve=1"
        cfg.max_solver_threads = 8
        sh = shared_options(cfg)
        spoke = self._spoke_dict_from(sh)
        apply_solver_specs("lagrangian", spoke, cfg)
        opts = spoke["opt_kwargs"]["options"]
        self.assertEqual(opts["iter0_solver_options"].get("threads"), 8)
        self.assertEqual(opts["iterk_solver_options"].get("threads"), 8)
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 0),
            opts["iter0_solver_options"],
        )
        self.assertEqual(
            fold_solver_options_layers(opts["solver_options_layers"], 1),
            opts["iterk_solver_options"],
        )


if __name__ == "__main__":
    unittest.main()
