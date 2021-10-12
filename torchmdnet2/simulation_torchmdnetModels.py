#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:38:18 2021

@author: schreibef98
"""

import torch
import numpy as np

from torch_geometric.data import Data, DataLoader

import os
import time
import warnings

from .utils import tqdm
from .simulation import Simulation, PTSimulation

from torchmd.run import setup
from torchmd.wrapper import Wrapper



class Simulation_(Simulation):
    """Simulate an artificial trajectory from a CGnet.
    
    If friction and masses are provided, Langevin dynamics are used (see also
    Notes). The scheme chosen is BAOA(F)B, where
        B = deterministic velocity update
        A = deterministic position update
        O = stochastic velocity update
        F = force calculation (i.e., from the cgnet)
    
    Where we have chosen the following implementation so as to only calculate
    forces once per timestep:
        F = - grad( U(X_t) )
        [BB] V_(t+1) = V_t + dt * F/m
        [A] X_(t+1/2) = X_t + V * dt/2
        [O] V_(t+1) = V_(t+1) * vscale + dW_t * noisescale
        [A] X_(t+1) = X_(t+1/2) + V * dt/2
    
    Where:
        vscale = exp(-friction * dt)
        noisecale = sqrt(1 - vscale * vscale)
    
    The diffusion constant D can be back-calculated using the Einstein relation
        D = 1 / (beta * friction)
    
    Initial velocities are set to zero with Gaussian noise.
    
    If friction is None, this indicates Langevin dynamics with *infinite*
    friction, and the system evolves according to overdamped Langevin
    dynamics (i.e., Brownian dynamics) according to the following stochastic
    differential equation:
    
        dX_t = - grad( U( X_t ) ) * D * dt + sqrt( 2 * D * dt / beta ) * dW_t
    
    for coordinates X_t at time t, potential energy U, diffusion D,
    thermodynamic inverse temperature beta, time step dt, and stochastic Weiner
    process W. The choice of Langevin dynamics is made because CG systems
    possess no explicit solvent, and so Brownian-like collisions must be
    modeled indirectly using a stochastic term.
    
    Parameters
    ----------
    model : cgnet.network.CGNet() instance
        Trained model used to generate simulation data
    initial_coordinates : np.ndarray or torch.Tensor
        Coordinate data of dimension [n_simulations, n_atoms, n_dimensions].
        Each entry in the first dimension represents the first frame of an
        independent simulation.
    embeddings : np.ndarray or None (default=None)
        Embedding data of dimension [n_simulations, n_beads]. Each entry
        in the first dimension corresponds to the embeddings for the
        initial_coordinates data. If no embeddings, use None.
    dt : float (default=5e-4)
        The integration time step for Langevin dynamics. Units are determined
        by the frame striding of the original training data simulation
    beta : float (default=1.0)
        The thermodynamic inverse temperature, 1/(k_B T), for Boltzman constant
        k_B and temperature T. The units of k_B and T are fixed from the units
        of training forces and settings of the training simulation data
        respectively
    friction : float (default=None)
        If None, overdamped Langevin dynamics are used (this is equivalent to
        "infinite" friction). If a float is given, Langevin dynamics are
        utilized with this (finite) friction value (sometimes referred to as
        gamma)
    masses : list of floats (default=None)
        Only relevant if friction is not None and (therefore) Langevin dynamics
        are used. In that case, masses must be a list of floats where the float
        at mass index i corresponds to the ith CG bead.
    diffusion : float (default=1.0)
        The constant diffusion parameter D for overdamped Langevin dynamics
        *only*. By default, the diffusion is set to unity and is absorbed into
        the dt argument. However, users may specify separate diffusion and dt
        parameters in the case that they have some estimate of the CG
        diffusion
    save_forces : bool (defalt=False)
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_potential : bool (default=False)
        Whether to save potential at the same saved interval as the simulation
        coordinates
    length : int (default=100)
        The length of the simulation in simulation timesteps
    save_interval : int (default=10)
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    random_seed : int or None (default=None)
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device : torch.device (default=torch.device('cpu'))
        Device upon which simulation compuation will be carried out
    export_interval : int (default=None)
        If not None, .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_potential
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
    log_interval : int (default=None)
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type : 'print' or 'write' (default='write')
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename : string (default=None)
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.
    
    Notes
    -----
    Long simulation lengths may take a significant amount of time.
    
    Langevin dynamics simulation velocities are currently initialized from
    zero. You should probably remove the beginning part of the simulation.
    
    Any output files will not be overwritten; the presence of existing files
    will cause an error to be raised.
    
    Langevin dynamics code based on:
    https://github.com/choderalab/openmmtools/blob/master/openmmtools/integrators.py
    
    Checks are broken into two methods: one (_input_model_checks()) that deals
    with checking components of the input models and their architectures and another
    (_input_option_checks()) that deals with checking options pertaining to the
    simulation, such as logging or saving/exporting options. This is done for the
    following reasons:
    
    		1) If cgnet.network.Simluation is subclassed for multiple or specific
        model types, the _input_model_checks() method can be overridden without
        repeating a lot of code. As an example, see
        cgnet.network.MultiModelSimulation, which uses all of the same
        simulation options as cgnet.network.Simulation, but overides
        _input_model_checks() in order to perform the same model checks as
        cgnet.network.Simulation but for more than one input model.
    
    		2) If cgnet.network.Simulation is subclassed for different simulation
        schemes with possibly different/additional simulation
        parameters/options, the _input_option_checks() method can be overriden
        without repeating code related to model checks. For example, one might
        need a simulation scheme that includes an external force that is
        decoupled from the forces predicted by the model.
    
    """
    def __init__(self, args, initial_coordinates, embeddings, dt=5e-4,
                 beta=1.0, friction=None, masses=None, diffusion=1.0,
                 save_forces=False, save_potential=False, length=100,
                 save_interval=10, random_seed=None,
                 device=torch.device('cpu'),
                 export_interval=None, log_interval=None,
                 log_type='write', filename=None, batch_size=10):
        
        self.args = args
        self.mol, self.system, self.forces = setup(self.args)
        #self.wrapper = Wrapper(self.mol.numAtoms, self.mol.bonds if len(self.mol.bonds) else None, device)

        self.initial_coordinates = initial_coordinates
        self.embeddings = embeddings

        self.friction = friction
        self.masses = masses

        self.n_sims = self.initial_coordinates.shape[0]
        self.n_beads = self.initial_coordinates.shape[1]
        self.n_dims = self.initial_coordinates.shape[2]

        self.save_forces = save_forces
        self.save_potential = save_potential
        self.length = length
        self.save_interval = save_interval

        self.dt = dt
        self.diffusion = diffusion
        self.beta = beta

        self.device = device
        self.export_interval = export_interval
        self.log_interval = log_interval
        self.batch_size = batch_size
        if log_type not in ['print', 'write']:
            raise ValueError(
                "log_type can be either 'print' or 'write'"
            )
        self.log_type = log_type
        self.filename = filename
        # Here, we check to make sure input options for the simulation
        # are acceptable. Note that these checks are separated from
        # the input model checks in _input_model_checks() for ease in
        # subclassing. See class notes for more information.
        self._input_option_checks()

        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)
        self.random_seed = random_seed

        self._simulated = False
        
        
    def calculate_potential_and_forces(self, x_old):
        #forces = torch.zeros_like(x_old, dtype=torch.float64).to(self.device)
        x_old = x_old.detach()
        epot = self.forces.compute(x_old, self.system.box, self.system.forces)
        
        return torch.Tensor(epot), self.system.forces

   








class PTSimulation_(PTSimulation):
    def __init__(self, args, initial_coordinates, embeddings, dt=5e-4,
                 exchange_interval=200,
                 betas=1.0, friction=None, masses=None, diffusion=1.0,
                 save_forces=False, save_potential=False, length=100,
                 save_interval=10, random_seed=None,
                 device=torch.device('cpu'),
                 export_interval=None, log_interval=None,
                 log_type='write', filename=None, batch_size=10,
                 test_force_field=False):

        self.args = args
        self.mol, self.system, self.forces = setup(self.args)
        #self.wrapper = Wrapper(self.mol.numAtoms, self.mol.bonds if len(self.mol.bonds) else None, device)
        
        self.friction = friction
        self.masses = masses

        self.save_forces = save_forces
        self.save_potential = save_potential
        self.length = length
        self.save_interval = save_interval

        self.dt = dt
        self.diffusion = diffusion
        self.betas = betas
        self.device = device
        self.export_interval = export_interval
        self.log_interval = log_interval
        self.batch_size = batch_size
        if log_type not in ['print', 'write']:
            raise ValueError(
                "log_type can be either 'print' or 'write'"
            )
        self.log_type = log_type
        self.filename = filename
        self.test_force_field = test_force_field

        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)
        self.random_seed = random_seed

        self._simulated = False
        
        ####################################
        # additional stuff for PT simulation
        ####################################
        
        # checking customized inputs
        betas = np.array(betas)
        if len(betas.shape) != 1 or betas.shape[0] <= 1:
            raise ValueError('betas must have shape (n_replicas,), where '
                             'n_replicas > 1.')
        self._betas = betas
        if type(exchange_interval) is not int or exchange_interval < 0:
            raise ValueError('exchange_interval must be a positive integer.')
        self.exchange_interval = exchange_interval

        # identify number of replicas
        self.n_replicas = len(self._betas)

        # preparing initial coordinates for each replica
        if type(initial_coordinates) is torch.Tensor:
            initial_coordinates = initial_coordinates.detach().cpu().numpy()
        new_initial_coordinates = np.concatenate([initial_coordinates] * 
                                                 self.n_replicas)
        
        self.initial_coordinates = new_initial_coordinates
        
        self.n_sims = self.initial_coordinates.shape[0]
        self.n_beads = self.initial_coordinates.shape[1]
        self.n_dims = self.initial_coordinates.shape[2]
        
        # preparing embeddings for each replica
        if embeddings is not None:
            if type(embeddings) is torch.Tensor:
                embeddings = embeddings.detach().cpu().numpy()
            embeddings = np.concatenate([embeddings] * self.n_replicas)
            embeddings = torch.tensor(embeddings, dtype=torch.int64
                                     ).to(self.device)
        self.embeddings = embeddings
        
        # set up betas for simulation
        self._n_indep = len(initial_coordinates)
        self._betas_x = np.repeat(self._betas, self._n_indep)
        self._betas_for_simulation = torch.tensor(self._betas_x[:, None, None],
                                                  dtype=torch.float32
                                                 ).to(self.device)
        # for replica exchange pair proposing
        self._propose_even_pairs = True
        even_pairs = [(i, i + 1) for i in np.arange(self.n_replicas)[:-1:2]]
        odd_pairs = [(i, i + 1) for i in np.arange(self.n_replicas)[1:-1:2]]
        if len(odd_pairs) == 0:
            odd_pairs = even_pairs
        pair_a = []
        pair_b = []
        for pair in even_pairs:
            pair_a.append(np.arange(self._n_indep) + pair[0] * self._n_indep)
            pair_b.append(np.arange(self._n_indep) + pair[1] * self._n_indep)
        self._even_pairs = [np.concatenate(pair_a), np.concatenate(pair_b)]
        pair_a = []
        pair_b = []
        for pair in odd_pairs:
            pair_a.append(np.arange(self._n_indep) + pair[0] * self._n_indep)
            pair_b.append(np.arange(self._n_indep) + pair[1] * self._n_indep)
        self._odd_pairs = [np.concatenate(pair_a), np.concatenate(pair_b)]
        
        
        # Moved here in PTSimulation
        # Here, we check to make sure input options for the simulation
        # are acceptable. Note that these checks are separated from
        # the input model checks in _input_model_checks() for ease in
        # subclassing. See class notes for more information.
        self._input_option_checks()
        
    def calculate_potential_and_forces(self, x_old):
        #forces = torch.zeros_like(x_old, dtype=torch.float64).to(self.device)
        x_old = x_old.detach()
        epot = self.forces.compute(x_old, self.system.box, self.system.forces)
        
        return torch.Tensor(epot), self.system.forces