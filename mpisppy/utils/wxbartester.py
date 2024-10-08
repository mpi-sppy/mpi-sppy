###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
''' Driver script to test the w/xbar read/write extensions using UC 
'''

import mpisppy.tests.examples.uc.uc_funcs as uc_funcs
# this version is locked to three scenarios

from mpisppy.utils.wxbarreader import WXBarReader

from mpisppy.opt.ph import PH

def nonsense(arg1, arg2, arg3): # Empty scenario_denouement
    pass

def read_test():
    scen_count          = 3
    scenario_creator    = uc_funcs.pysp2_callback
    scenario_denouement = nonsense
    scenario_rhosetter  = uc_funcs.scenario_rhos

    PH_options = {
        'solver_name': 'gurobi',
        'PHIterLimit': 2,
        'defaultPHrho': 1,
        'convthresh': 1e-6,
        'verbose': False,
        'display_timing': False,
        'display_progress': False,
        'iter0_solver_options': dict(),
        'iterk_solver_options': dict(),
        'init_W_fname': 'david/weights.csv', # Path to the weight files
        'init_separate_W_files': False,
        'init_Xbar_fname': 'david/somexbars.csv',
        'extensions':WXBarReader,
        'rho_setter':scenario_rhosetter,
    }

    names = ['Scenario' + str(i+1) for i in range(scen_count)]

    ph = PH(PH_options, names, scenario_creator, scenario_denouement)

    conv, obj, bound = ph.ph_main()

def write_test():
    scen_count          = 3
    scenario_creator    = uc_funcs.pysp2_callback
    scenario_denouement = nonsense
    scenario_rhosetter  = uc_funcs.scenario_rhos

    PH_options = {
        'solver_name': 'gurobi',
        'PHIterLimit': 2,
        'defaultPHrho': 1,
        'convthresh': 1e-6,
        'verbose': False,
        'display_timing': False,
        'display_progress': False,
        'iter0_solver_options': dict(),
        'iterk_solver_options': dict(),
        'W_fname': 'david/weights.csv',
        'separate_W_files': False,
        'Xbar_fname': 'somexbars.csv',
        'extensions':WXBarReader,
        'rho_setter':scenario_rhosetter,
    }

    names = ['Scenario' + str(i+1) for i in range(scen_count)]

    ph = PH(PH_options, names, scenario_creator, scenario_denouement)

    conv, obj, bound = ph.ph_main()

if __name__=='__main__':
    read_test()
