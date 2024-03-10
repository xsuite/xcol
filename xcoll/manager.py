# copyright ############################### #
# This file is part of the Xcoll Package.   #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .beam_elements import BaseCollimator, BlackAbsorber, EverestCollimator, EverestCrystal, _all_collimator_types, element_classes
from .colldb import CollimatorDatabase
from .impacts import CollimatorImpacts
from .scattering_routines.everest.materials import SixTrack_to_xcoll, CrystalMaterial

import xobjects as xo
import xpart as xp
import xtrack as xt


# Logic of buffers:
#    A buffer is a continuous stream of memory, containing a bunch of xobjects.
#    It is an architecture-invariant way to work with pointers, as the pointers to the individual
#    xobjects are stored as offsets to the start of the buffer.
#    The tracker manages three buffers: one for the line and all elements, one for the particles,
#    and an io_buffer for record logging.

class CollimatorManager:

    _init_vars = ['_colldb', 'line', 'beam', 'capacity', 'record_impacts', '_context', '_buffer', 'io_buffer', '_part_buffer']
    _init_var_defaults = {'_colldb': None, 'beam': None, 'capacity': 1e6, 'record_impacts': False, \
                          '_context': None, '_buffer': None, 'io_buffer': None, '_part_buffer': None}

    # -------------------------------
    # ------ Loading functions ------
    # -------------------------------

    @classmethod
    def from_yaml(cls, file, **kwargs):
        if '_colldb' in kwargs.keys():
            raise ValueError("Cannot load CollimatorDatabase file and specify '_colldb' argument in loader!")
        kwargs_colldb  = {key: val for key, val in kwargs.items() if key in CollimatorDatabase._init_vars}
        kwargs_manager = {key: val for key, val in kwargs.items() if key in cls._init_vars}
        colldb = CollimatorDatabase.from_yaml(file, **kwargs_colldb)
        return cls(_colldb=colldb, **kwargs_manager)

    @classmethod
    def from_json(cls, file, **kwargs):
        if '_colldb' in kwargs.keys():
            raise ValueError("Cannot load CollimatorDatabase file and specify '_colldb' argument in loader!")
        kwargs_colldb  = {key: val for key, val in kwargs.items() if key in CollimatorDatabase._init_vars}
        kwargs_manager = {key: val for key, val in kwargs.items() if key in cls._init_vars}
        colldb = CollimatorDatabase.from_json(file, **kwargs_colldb)
        return cls(_colldb=colldb, **kwargs_manager)

    @classmethod
    def from_dict(cls, file, **kwargs):
        if '_colldb' in kwargs.keys():
            raise ValueError("Cannot load CollimatorDatabase file and specify '_colldb' argument in loader!")
        kwargs_colldb  = {key: val for key, val in kwargs.items() if key in CollimatorDatabase._init_vars}
        kwargs_manager = {key: val for key, val in kwargs.items() if key in cls._init_vars}
        colldb = CollimatorDatabase.from_dict(file, **kwargs_colldb)
        return cls(_colldb=colldb, **kwargs_manager)

    @classmethod
    def from_SixTrack(cls, file, **kwargs):
        if '_colldb' in kwargs.keys():
            raise ValueError("Cannot load CollimatorDatabase file and specify '_colldb' argument in loader!")
        kwargs_colldb  = {key: val for key, val in kwargs.items() if key in CollimatorDatabase._init_vars}
        kwargs_manager = {key: val for key, val in kwargs.items() if key in cls._init_vars}
        colldb = CollimatorDatabase.from_SixTrack(file, **kwargs_colldb)
        return cls(_colldb=colldb, **kwargs_manager)


    def __init__(self, **kwargs):
        # Get all arguments
        for var in self._init_vars:
            if var in self._init_var_defaults:
                kwargs.setdefault(var, self._init_var_defaults[var])
            elif var not in kwargs.keys():
                raise ValueError(f"CollimatorManager is missing required argument '{var}'!")

        colldb = kwargs['_colldb']
        if not isinstance(colldb, CollimatorDatabase):
            # TODO allow None as empty database
            raise ValueError("The variable '_colldb' needs to be an xcoll CollimatorDatabase object!")
        else:
            self.colldb = colldb

        line = kwargs['line']
        beam = kwargs['beam']
        if isinstance(beam, str):
            beam = int(beam[-1])
        if not isinstance(line, xt.Line):
            raise ValueError("The variable 'line' needs to be an xtrack Line object!")
        else:
            self.line = line
        self.line._needs_rng = True  # TODO not needed if only BlackAbsorbers
        if beam is not None and beam > 1:
            self._line_is_reversed = True
        else:
            self._line_is_reversed = False
        self._machine_length = self.line.get_length()

        # Create _buffer, _context, and _io_buffer
        _buffer  = kwargs['_buffer']
        _context = kwargs['_context']
        if _buffer is None:
            if _context is None:
                _context = xo.ContextCpu()
            _buffer = _context.new_buffer()
        elif _context is not None and _buffer.context != _context:
            raise ValueError("The provided buffer and context do not match! "
                             + "Make sure the buffer is generated inside the provided context, or alternatively, "
                             + "only pass one of _buffer or _context.")
        self._buffer = _buffer
        self._part_buffer  = kwargs['_part_buffer']
        if self._part_buffer is not None and self._part_buffer.context != _context:
            raise ValueError("The provided particle buffer and context do not match! "
                             + "Make sure the particle buffer is generated inside the provided context.")

        # TODO: currently capacity is only for io_buffer (hence for impacts).
        # Do we need it in the _buffer as well?
        self._capacity = int(kwargs['capacity'])
        io_buffer = kwargs['io_buffer']
        if io_buffer is None:
            io_buffer = xt.new_io_buffer(_context=self._buffer.context, capacity=self.capacity)
        elif self._buffer.context != io_buffer._context:
            raise ValueError("The provided io_buffer lives on a different context than the buffer!")
        self._io_buffer = io_buffer

        # Initialise impacts table
        self._recording_elements = []
        self.record_impacts = kwargs['record_impacts']
        self._impacts = None

        # Variables for lossmap
        self._lossmap = None
        self._summary = None
        self._part    = None


    def __getitem__(self, name):
        return self.colldb[name]


    @property
    def machine_length(self):
        return self._machine_length

    @property
    def impacts(self):
        return self._impacts

    @property
    def record_impacts(self):
        return self._record_impacts

    @record_impacts.setter
    def record_impacts(self, record_impacts):
        if record_impacts is False or record_impacts is None:
            record_stop  = [self.line[name] for name in self._recording_elements]
            record_start = False
        elif record_impacts is True:
            record_stop  = []
            record_start = True
        else:
            if not hasattr(record_impacts, '__iter__') or isinstance(record_impacts, str):
                record_impacts = [record_impacts]
            record_stop  = [self.line[name] for name in set(self._recording_elements) - set(record_impacts)]
            record_start = record_impacts

        # We stop logging these elements
        if record_stop:
            if self.line.tracker is not None:
                self.line.tracker._check_invalidated()
            xt.stop_internal_logging(elements=record_stop)
            # Removed the stopped collimators from list of logged elements
            self._recording_elements = list(set(self._recording_elements) - set(record_stop))

        # We will start logging these elements
        self._record_impacts = record_start
        self._set_record_impacts()

    def _set_record_impacts(self):
        record_impacts = self._record_impacts
        if record_impacts is True:
            record_impacts = self.collimator_names
        if record_impacts is False or record_impacts is None:
            record_impacts = []
        if not hasattr(record_impacts, '__iter__') or isinstance(record_impacts, str):
            record_impacts = [record_impacts]
        if record_impacts and np.all([isinstance(self.line[elem], BaseCollimator) for elem in record_impacts]):
            for name in record_impacts:
                if name not in self.line.element_names:
                    raise ValueError(f"Trying to initialise impact table, but collimator {name} not found in line!")
            elements_to_record = list(set(record_impacts) - set(self._recording_elements))
            elements_to_stop = list(set(self._recording_elements) - set(record_impacts))
            if elements_to_stop:
                # These should have been stopped by the setter function above
                raise ValueError("Some elements are recording but are not supposed to!")
            # Any new elements that need recording but aren't recording yet?
            if elements_to_record:
                elements_to_record = [self.line[name] for name in elements_to_record]
                if self.impacts is None:
                    self._impacts = xt.start_internal_logging(io_buffer=self._io_buffer, capacity=self.capacity, \
                                                              elements=elements_to_record)
                else:
                    xt.start_internal_logging(io_buffer=self._io_buffer, capacity=self.capacity, \
                                              record=self.impacts, elements=elements_to_record)
                self._recording_elements = record_impacts
            self.impacts._coll_ids = {self.line.element_names.index(name): name for name in self._recording_elements}

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, capacity):
        capacity = int(capacity)
        if capacity < self.capacity:
            raise NotImplementedError("Shrinking of capacity not yet implemented!")
        elif capacity == self.capacity:
            return
        else:
            self._io_buffer.grow(capacity-self.capacity)
            if self.impacts is not None:
                # TODO: increase capacity of iobuffer AND of _impacts
                raise NotImplementedError

    @property
    def collimator_names(self):
        # TODO: should only sort whenever collimators are added to colldb
        # or when positions are updated
        db = self.colldb._colldb
#         names = list(db[db.active==True].index)
        names = list(db.index)
        if not np.any([s is None for s in db.s_center]):
            names.sort(key=lambda nn: db.loc[nn, 's_center'])
        return names

    @property
    def s_front(self):
        return self.colldb.s_center - self.colldb.active_length/2 - self.colldb.inactive_front

    @property
    def s_active_front(self):
        return self.colldb.s_center - self.colldb.active_length/2

    @property
    def s_center(self):
        return self.colldb.s_center

    @property
    def s_active_back(self):
        return self.colldb.s_center + self.colldb.active_length/2

    @property
    def s_back(self):
        return self.colldb.s_center + self.colldb.active_length/2 + self.colldb.inactive_back

    def install_black_absorbers(self, names=None, *, verbose=False):
        if names is None:
            names = self.collimator_names
        def install_func(thiscoll, name):
            return BlackAbsorber(
                    inactive_front=thiscoll['inactive_front'],
                    inactive_back=thiscoll['inactive_back'],
                    active_length=thiscoll['active_length'],
                    angle=[thiscoll['angle_L'],thiscoll['angle_R']],
                    active=False,
                    _tracking=False,
                    _buffer=self._buffer
                   )
        self._install_collimators(names, install_func=install_func, verbose=verbose)


    def install_everest_collimators(self, names=None, *, verbose=False):
        if names is None:
            names = self.collimator_names
        names = list(names) # Dataframe does not like to be indexed with a set
        df = self.colldb._colldb.loc[names]
        df_coll = df[[c is None for c in df.crystal]]
        df_cry  = df[[c is not None for c in df.crystal]]
        # Do the installations (start with crystals to avoid recompilation)
        if len(df_cry) > 0:
            def install_func(thiscoll, name):
                material = SixTrack_to_xcoll[thiscoll['material']]
                if len(material) < 2:
                    raise ValueError(f"Could not find crystal material definition from variable {thiscoll['material']}!")
                material = material[1]
                if not isinstance(material, CrystalMaterial):
                    raise ValueError(f"The material {material.name} is not a Crystalmaterial!")
                return EverestCrystal(
                        inactive_front=thiscoll['inactive_front'],
                        inactive_back=thiscoll['inactive_back'],
                        active_length=thiscoll['active_length'],
                        angle=[thiscoll['angle_L'],thiscoll['angle_R']],
                        material=material,
                        active=False,
                        _tracking=False,
                        _buffer=self._buffer
                       )
            self._install_collimators(df_cry.index.values, install_func=install_func, verbose=verbose)
        if len(df_coll) > 0:
            def install_func(thiscoll, name):
                return EverestCollimator(
                        inactive_front=thiscoll['inactive_front'],
                        inactive_back=thiscoll['inactive_back'],
                        active_length=thiscoll['active_length'],
                        angle=[thiscoll['angle_L'],thiscoll['angle_R']],
                        material=SixTrack_to_xcoll[thiscoll['material']][0],
                        active=False,
                        _tracking=False,
                        _buffer=self._buffer
                       )
            self._install_collimators(df_coll.index.values, install_func=install_func, verbose=verbose)


    def _install_collimators(self, names, *, install_func, verbose, support_legacy_elements=False):
        # Check that collimator marker exists in Line and CollimatorDatabase,
        # and that tracker is not yet built
        # TODO: need check that all collimators have aperture before and after
        line = self.line
        if not hasattr(names, '__iter__') or isinstance(names, str):
            names = [names]
        df = self.colldb._colldb
        mask = df.index.isin(names)
        for name in names:
            if name not in line.element_names:
                raise Exception(f"Collimator {name} not found in line!")
            elif name not in self.collimator_names:
                raise Exception(f"Warning: Collimator {name} not found in CollimatorDatabase!...")
        if self.tracker_ready:
            raise Exception("Tracker already built!\nPlease install collimators before building "
                          + "tracker!")

        # Loop over collimators to install
        for name in names:

            # Get s positions
            # This cannot go outside the loop as the indices will change!
            ss = line.get_s_position()

            # Get the settings from the CollimatorDatabase
            thiscoll = df.loc[name]
            idx = line.element_names.index(name)
            # Create the collimator element
            newcoll = install_func(thiscoll, name)
            collimator_class = newcoll.__class__

            # Check that collimator is not installed as different type
            # TODO: automatically replace collimator type and print warning
            if isinstance(line[name], tuple(_all_collimator_types - {collimator_class})):
                raise ValueError(f"Trying to install {name} as {collimator_class.__name__},"
                               + f" but it is already installed as {type(line[name]).__name__}!\n"
                               + f"Please reconstruct the line.")

            # Check that collimator is not installed previously
            elif isinstance(line[name], collimator_class):
                if df.loc[name,'collimator_type'] != collimator_class.__name__:
                    raise Exception(f"Something is wrong: Collimator {name} already installed in "
                                  + f"line as {collimator_class.__name__} element, but registered "
                                  + f"in CollimatorDatabase as {df.loc[name, 'collimator_type']}. "
                                  + f"Please reconstruct the line.")
                if verbose: print(f"Collimator {name} already installed. Skipping...")
                continue

            # TODO: only allow Marker elements, no Drifts!!
            #       How to do this with importing a line for MAD-X or SixTrack...?
            elif not isinstance(line[name], (xt.Marker, xt.Drift)) and not support_legacy_elements:
                raise ValueError(f"Trying to install {name} as {collimator_class.__name__},"
                               + f" but the line element to replace is not an xtrack.Marker "
                               + f"(or xtrack.Drift)!\nPlease check the name, or correct the "
                               + f"element.")

            if verbose: print(f"Installing {name:20} as {collimator_class.__name__}")
            # Update the position and type in the CollimatorDatabase
            df.loc[name,'s_center'] = ss[idx]
            df.loc[name,'collimator_type'] = collimator_class.__name__

            # Find apertures and store them
            # TODO: same with cryotanks for FLUKA
            # TODO: use compound info  ->  need full collimator info from MADX
            # TODO: this is all very hacky....
            aper_before = {}
            aper_after = {}
            if f'{name}_mken' in line.element_names\
            and f'{name}_mkex'in line.element_names:
                # TODO what with transformations? How to shift them in s if different?
                aper_before = {nn.replace('mken', 'upstream'): line[nn].copy()
                               for nn in line.element_names if nn.startswith(f'{name}_mken_aper')}
                aper_after  = {nn.replace('mkex', 'downstream'): line[nn].copy()
                               for nn in line.element_names if nn.startswith(f'{name}_mkex_aper')}
            if len(aper_before) == 0:
                # TODO what with transformations? How to shift them in s from centre to start/end?
                aper_before = {nn.replace('_aper', '_upstream_aper'): line[nn].copy()
                               for nn in line.element_names if nn.startswith(f'{name}_aper')}
            if len(aper_after) == 0:
                aper_after  = {nn.replace('_aper', '_downstream_aper'): line[nn].copy()
                               for nn in line.element_names if nn.startswith(f'{name}_aper')}
            if len(aper_before) == 0 or len(aper_after) == 0:
                print(f"Warning: No aperture found for collimator {name}!")

            # Remove stuff at location of collimator
            l = thiscoll['active_length']
            to_remove = []
            i = idx - 1
            # We remove everything between the beginning and end of the collimator except drifts
            while ss[i] >= ss[idx] - l/2:
                el = line[i]
                nn = line.element_names[i]
                if el.__class__.__name__ == 'Drift':
                    i -= 1
                    continue
                if hasattr(el, 'length') and el.length > 0:
                    raise ValueError(f"Found active element {nn} with length "
                                   + f"{el.length} at location inside collimator!")
                # I don't like this class selection...
                if not el.__class__.__name__ in ['Marker', 'SRotation', \
                                       'YRotation', 'XRotation', 'XYShift'] \
                and not el.__class__.__name__.startswith('Limit'):
                    print(f"Warning: Removed active element {nn} "
                        + f"at location inside collimator!")
                to_remove.append(nn)
                i -= 1
            i = idx + 1
            while ss[i] <= ss[idx] + l/2:
                el = line[i]
                nn = line.element_names[i]
                if el.__class__.__name__ == 'Drift':
                    i += 1
                    continue
                if hasattr(el, 'length') and el.length > 0:
                    raise ValueError(f"Found active element {nn} with length "
                                   + f"{el.length} at location inside collimator!")
                # I don't like this class selection...
                if not el.__class__.__name__ in ['Marker', 'SRotation', \
                                       'YRotation', 'XRotation', 'XYShift'] \
                and not el.__class__.__name__.startswith('Limit'):
                    print(f"Warning: Removed active element {line.element_names[i]} "
                        + f"at location inside collimator!")
                to_remove.append(nn)
                i += 1
            for nn in to_remove:
                # TODO: need to update Compounds
                line.element_names.remove(nn)
                line.element_dict.pop(nn)

            # Do the installation
            s_install = df.loc[name,'s_center'] - thiscoll['active_length']/2 \
                            - thiscoll['inactive_front']
            line.insert_element(element=newcoll, name=name, at_s=s_install)

            # Reinstall apertures
            for aper, el in aper_before.items():
                # TODO: need to update Compounds
                line.insert_element(element=el, name=aper, index=name)
            for aper, el in reversed(aper_after.items()):
                # Reversed because of index+1
                # TODO: need to update Compounds
                line.insert_element(element=el, name=aper,
                                    index=line.element_names.index(name)+1)

        self._set_record_impacts()


    @property
    def installed(self):
        return not any([coll is None for coll in self.colldb.collimator_type])


    def align_collimators_to(self, align):
        if not self.installed:
            raise ValueError("Some collimators have not yet been installed.\n"
                             + "Please install all collimators before aligning the collimators.")
        self.colldb.align_to = align

    @property
    def aligned(self):
        return not np.any(
                    [x is None for x in self.colldb._colldb.s_align_front]
                    + [ x is None for x in self.colldb._colldb.s_align_back]
                )


    def build_tracker(self, **kwargs):
        kwargs.setdefault('_buffer', self._buffer)
        kwargs.setdefault('io_buffer', self._io_buffer)
        if kwargs['_buffer'] != self._buffer:
            raise ValueError("Cannot build tracker with different buffer than the CollimatorManager buffer!")
        if kwargs['io_buffer'] != self._io_buffer:
            raise ValueError("Cannot build tracker with different io_buffer than the CollimatorManager io_buffer!")
        if '_context' in kwargs and kwargs['_context'] != self._buffer.context:
            raise ValueError("Cannot build tracker with different context than the CollimatorManager context!")
        self.line.build_tracker(**kwargs)
        self._set_record_impacts()

    @property
    def tracker_ready(self):
        return self.line is not None and self.line.tracker is not None


    def _compute_optics(self, recompute=False):
        if not self.tracker_ready:
            raise Exception("Please build tracker before computing the optics for the openings!")
        line = self.line

        if recompute or not self.colldb._optics_is_ready:
            # TODO: does this fail on Everest? Can the twiss be calculated at the center of the collimator for everest?
#             pos = { *self.s_active_front, *self.s_center, *self.s_active_back }
            pos = list({ *self.s_active_front, *self.s_active_back })
            tw = line.twiss(at_s=pos)
#             tw = tracker.twiss()
            self.colldb._optics = pd.concat([
                                    self.colldb._optics,
                                    pd.DataFrame({
                                            opt: tw[opt] for opt in self.colldb._optics.columns
#                                     opt: [ np.array(tw[opt])[abs(tw['s']-thispos) < 1e-12][0] for thispos in pos ]
#                                         for opt in self.colldb._optics.columns
                                    }, index=pos)
                                ])
            self.colldb.gamma_rel = line.particle_ref._xobject.gamma0[0]


    # The variable 'gaps' is meant to specify temporary settings that will overrule the CollimatorDatabase.
    # As such, its settings will be applied to the collimator elements in the line, but not
    # written to the CollimatorDatabase. Hence two successive calls to set_openings will not be combined,
    # and only the last call will be applied to the line.
    # The variable 'to_parking' will send all collimators that are not listed in 'gaps' to parking.
    # Similarily, the variable 'full_open' will set all openings of the collimators that are not
    # listed in 'gaps' to 1m.
    def set_openings(self, gaps={}, *, recompute_optics=False, to_parking=False, full_open=False, support_legacy_elements=False):
        if not self.tracker_ready:
            raise Exception("Please build tracker before setting the openings!")
        colldb = self.colldb
        if not self.installed:
            raise ValueError("Some collimators have not yet been installed.\n"
                             + "Please install all collimators before setting the openings.")
        if to_parking and full_open:
            raise ValueError("Cannot send collimators to parking and open them fully at the same time!")
        if not self.aligned:
            self.align_collimators_to('front')

        gaps_OLD = colldb.gap
        names = self.collimator_names
        # Override gap if sending to parking
        if to_parking:
            gaps = { **{ name: None for name in names }, **gaps }
        colldb.gap = gaps

        # Get the optics (to compute the opening)
        self._compute_optics(recompute=recompute_optics)
        if not self.colldb._optics_is_ready:
            raise Exception("Something is wrong: not all optics needed for the jaw openings are calculated!")

        # Configure collimators
        line = self.line
        for name in names:
            # Override openings if opening fully
            if full_open and name not in gaps.keys():
                line[name].active = False
            # Apply settings to element
            elif isinstance(line[name], BaseCollimator):
                line[name].ref_x  = colldb.x[name]
                line[name].ref_y  = colldb.y[name]
                line[name].angle  = colldb.angle[name]
                line[name].jaw_L = colldb._colldb.jaw_LU[name]
                line[name].jaw_R = colldb._colldb.jaw_RU[name]
                # TODO
                line[name].side   = colldb.side[name]
                line[name].active = colldb.active[name]
                if isinstance(line[name], (EverestCollimator, EverestCrystal)) or support_legacy_elements:
                    if colldb._colldb.crystal[name] is None:
                        line[name].material = SixTrack_to_xcoll[colldb.material[name]][0]
                    else:
                        line[name].material = SixTrack_to_xcoll[colldb.material[name]][1]
                if isinstance(line[name], EverestCrystal):
                    line[name].align_angle    = colldb._colldb.align_angle[name]
                    line[name].bending_radius = colldb._colldb.bending_radius[name]
                    line[name].xdim           = colldb._colldb.xdim[name]
                    line[name].ydim           = colldb._colldb.ydim[name]
                    line[name].thick          = colldb._colldb.thick[name]
                    line[name].miscut         = colldb._colldb.miscut[name]
                    line[name].lattice        = colldb._colldb.crystal[name]
            else:
                raise ValueError(f"Missing implementation for element type of collimator {name}!")
        colldb.gap = gaps_OLD

    @property
    def openings_set(self):
        # TODO: need to delete jaw positions if some parameters (that would influence it) are changed
        return not np.any(
                    [x is None for x in self.colldb._colldb.jaw_LU]
                    + [ x is None for x in self.colldb._colldb.jaw_RU]
                    + [ x is None for x in self.colldb._colldb.jaw_LD]
                    + [ x is None for x in self.colldb._colldb.jaw_RD]
                )

    def generate_2D_pencil_one_jaw(self, num_particles, collimator, plane, side, impact_parameter,
                                   pencil_spread, tw, front=True):
        coll = self.line[collimator]
        if plane == 'x':
            co_pencil     = coll.ref_x
        elif plane == 'y':
            co_pencil     = coll.ref_y
        else:
            raise ValueError("`plane` must be either 'x' or 'y'!")
        if side == '+':
            absolute_cut = coll.jaw_LU + co_pencil + impact_parameter + pencil_spread
        elif side == '-':
            absolute_cut = coll.jaw_RU + co_pencil - impact_parameter - pencil_spread
        else:
            raise ValueError("`side` must be either '+' or '-'!")
        optics = tw.rows[collimator:f'{collimator}%%1']
        alfa = optics[f'alf{plane}'][0] if front else optics[f'alf{plane}'][1]
        beta = optics[f'bet{plane}'][0] if front else optics[f'bet{plane}'][1]
        gama = optics[f'gam{plane}'][0] if front else optics[f'gam{plane}'][1]
        particle_emitt = absolute_cut**2 / beta



    def _generate_4D_pencil_one_jaw(self, num_particles, collimator, plane, side, impact_parameter,
                                    dr_sigmas, transverse_spread_sigma, match_at_s):

        line = self.line

        if plane == 'x':
            co_pencil     = line[collimator].ref_x
            co_transverse = line[collimator].ref_y
        else:
            co_pencil     = line[collimator].ref_y
            co_transverse = line[collimator].ref_x

        nemitt_x   = self.colldb.emittance[0]
        nemitt_y   = self.colldb.emittance[1]

        if side == '+':
            absolute_cut = line[collimator].jaw_LU + co_pencil + impact_parameter
        elif side == '-':
            absolute_cut = line[collimator].jaw_RU + co_pencil - impact_parameter

        # Collimator plane: generate pencil distribution
        pencil, p_pencil = xp.generate_2D_pencil_with_absolute_cut(num_particles,
                        plane=plane, absolute_cut=absolute_cut, dr_sigmas=dr_sigmas,
                        side=side, line=line, nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                        at_element=collimator, match_at_s=match_at_s
        )

        # Other plane: generate gaussian distribution in normalized coordinates
        transverse_norm   = np.random.normal(loc=co_transverse, scale=transverse_spread_sigma, size=num_particles)
        p_transverse_norm = np.random.normal(scale=transverse_spread_sigma, size=num_particles)

        return [pencil, p_pencil, transverse_norm, p_transverse_norm]


    def generate_pencil_on_collimator(self, collimator, num_particles, *, side='+-', impact_parameter=0, 
                                      pencil_spread=1e-6, transverse_impact_parameter=0.,
                                      transverse_spread_sigma=0.01, longitudinal=None,
                                      longitudinal_betatron_cut=None, sigma_z=7.61e-2):
        if not self.openings_set:
            raise ValueError("Need to set collimator openings before generating pencil distribution!")
        if not self.tracker_ready:
            raise Exception("Please build tracker before generating pencil distribution!")
        if transverse_impact_parameter != 0.:
            raise NotImplementedError

        # TODO: check collimator in colldb and installed!

        if self.colldb.side[collimator] == 'left':
            side = '+'
        if self.colldb.side[collimator] == 'right':
            side = '-'

        # Define the plane
        angle = self.colldb.angle[collimator]
        if abs(np.mod(angle-90,180)-90) < 1e-6:
            plane = 'x'
        elif abs(np.mod(angle,180)-90) < 1e-6:
            plane = 'y'
        else:
            raise NotImplementedError("Pencil beam on a skew collimator not yet supported!")

        nemitt_x   = self.colldb.emittance[0]
        nemitt_y   = self.colldb.emittance[1]

        # Is it converging or diverging?
        is_converging = self.colldb._optics.loc[self.s_active_front[collimator], 'alf' + plane ] > 0
        print(f"Collimator {collimator} is {'con' if is_converging else 'di'}verging.")
        if is_converging:
            # pencil at front of jaw
            match_at_s = self.s_active_front[collimator]
            sigma      = self.colldb._beam_size_front[collimator]
        else:
            # pencil at back of jaw
            match_at_s = self.s_active_back[collimator]
            sigma      = self.colldb._beam_size_back[collimator]

        dr_sigmas = pencil_spread/sigma

        # Generate 4D coordinates
        if side == '+-':
            num_plus = int(num_particles/2)
            num_min  = int(num_particles - num_plus)
            coords_plus = self._generate_4D_pencil_one_jaw(num_plus, collimator, plane, '+',
                                                           impact_parameter, dr_sigmas,
                                                           transverse_spread_sigma, match_at_s)
            coords_min  = self._generate_4D_pencil_one_jaw(num_min, collimator, plane, '-',
                                                           impact_parameter, dr_sigmas,
                                                           transverse_spread_sigma, match_at_s)
            coords      = [ [*c_plus, *c_min] for c_plus, c_min in zip(coords_plus, coords_min)]
        else:
            coords      = self._generate_4D_pencil_one_jaw(num_particles, collimator, plane, side,
                                                           impact_parameter, dr_sigmas,
                                                           transverse_spread_sigma, match_at_s)
        pencil            = coords[0]
        p_pencil          = coords[1]
        transverse_norm   = coords[2]
        p_transverse_norm = coords[3]

        # Longitudinal plane
        # TODO: make this more general, make this better
        if longitudinal is None:
            delta = 0
            zeta  = 0
        elif longitudinal == 'matched_dispersion':
            if longitudinal_betatron_cut is None:
                cut = 0
            else:
                cut = np.random.uniform(-longitudinal_betatron_cut, longitudinal_betatron_cut, num_particles)
            delta = self.generate_delta_from_dispersion(at_element=collimator, plane=plane, position_mm=pencil,
                                                        betatron_cut=cut)
            zeta  = 0
        elif longitudinal == 'bucket':
            zeta, delta = xp.generate_longitudinal_coordinates(
                    num_particles=num_particles, distribution='gaussian', sigma_z=sigma_z, line=self.line
            )
        elif not hasattr(longitudinal, '__iter__'):
            raise ValueError
        elif len(longitudinal) != 2:
            raise ValueError
        elif isinstance(longitudinal, str):
            raise ValueError
        elif isinstance(longitudinal, dict):
            zeta = longitudinal['zeta']
            delta = longitudinal['delta']
        else:
            zeta = longitudinal[0]
            delta = longitudinal[1]

        # Build the particles
        if plane == 'x':
            part = xp.build_particles(
                    x=pencil, px=p_pencil, y_norm=transverse_norm, py_norm=p_transverse_norm,
                    zeta=zeta, delta=delta, nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                    line=self.line, at_element=collimator, match_at_s=match_at_s,
                    _buffer=self._part_buffer
            )
        else:
            part = xp.build_particles(
                    x_norm=transverse_norm, px_norm=p_transverse_norm, y=pencil, py=p_pencil, 
                    zeta=zeta, delta=delta, nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                    line=self.line, at_element=collimator, match_at_s=match_at_s,
                    _buffer=self._part_buffer
            )

        part._init_random_number_generator()

        return part


    def generate_delta_from_dispersion(self, at_element, *, plane, position_mm, betatron_cut=0):
        line = self.line
        if line.tracker is None:
            raise ValueError("Need to build tracker first!")
        if not hasattr(betatron_cut, '__iter__'):
            if hasattr(position_mm, '__iter__'):
                betatron_cut = np.full_like(position_mm, betatron_cut)
        elif not hasattr(position_mm, '__iter__'):
            position_mm = np.full_like(betatron_cut, position_mm)
        elif len(position_mm) != len(betatron_cut):
            raise ValueError
        tw = line.twiss()
        if isinstance(at_element, str):
            idx = line.element_names.index(at_element)
        betagamma = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
        if plane == 'x':
            sigma = np.sqrt(tw.betx[idx]*self.colldb.emittance[0]/betagamma)
            delta = (position_mm - betatron_cut*sigma - tw.x[idx]) / tw.dx[idx]
        elif plane == 'y':
            sigma = np.sqrt(tw.bety[idx]*self.colldb.emittance[1]/betagamma)
            delta = (position_mm - betatron_cut*sigma - tw.y[idx]) / tw.dy[idx]
        else:
            raise ValueError("The variable 'plane' needs to be either 'x' or 'y'!")
        return delta


    def enable_scattering(self):
        # Prepare collimators for tracking
        for el in self.line.get_elements_of_type(element_classes)[0]:
            el.enable_scattering()
        self._set_record_impacts()

    def disable_scattering(self):
        # Prepare collimators for tracking
        for el in self.line.get_elements_of_type(element_classes)[0]:
            el.disable_scattering()

    @property
    def scattering_enabled(self):
        all_enabled  = np.all([self.line[coll]._tracking if hasattr(self.line[coll], '_tracking')
                               else True for coll in self.collimator_names])
        some_enabled = np.any([self.line[coll]._tracking if hasattr(self.line[coll], '_tracking')
                               else True  for coll in self.collimator_names])
        if some_enabled and not all_enabled:
            raise ValueError("Some collimators are enabled for tracking but not all! "
                           + "This should not happen.")
        return all_enabled


    def summary(self, part, weights=None, show_zeros=False, file=None, recompute=False):

        # We cache the result
        if (self._summary is None or self._part is None
            or not xt.line._dicts_equal(part.to_dict(), self._part)
            or recompute
           ):
            if weights is None:
                weights = np.ones(len(part.x))
            else:
                part.sort(interleave_lost_particles=True)
            self._part   = part.to_dict()
            coll_mask    = (part.state<=-330) & (part.state>=-340)
            coll_losses  = np.array([self.line.element_names[i] for i in part.at_element[coll_mask]])
            coll_loss_single = np.unique(coll_losses)
            coll_lengths = [self.line[nn].active_length for nn in self.collimator_names] 
            coll_pos     = [self.colldb.s_center[nn]    for nn in self.collimator_names]
            if self._line_is_reversed:
                coll_pos = [self.machine_length - s for s in coll_pos ]
            coll_types   = [self.line[nn].__class__.__name__  for nn in self.collimator_names]
            coll_weights = weights[coll_mask]
            nabs         = [coll_weights[coll_losses==nn].sum() for nn in self.collimator_names]

            self._summary = pd.DataFrame({
                        "collname": self.collimator_names,
                        "nabs":     nabs,
                        "length":   coll_lengths,
                        "s":        coll_pos,
                        "type":     coll_types
                      })

        if file is not None:
            with open(Path(file), 'w') as fid:
                fid.write(self._summary.__repr__())

        if show_zeros:
            return self._summary
        else:
            return self._summary[self._summary.nabs > 0]


    def create_weights_from_initial_state(part, function):
        if len(function) == 4:
            return function[0](part.x)*function[1](part.px)*\
                   function[2](part.y)*function[3](part.py)
        elif len(function) == 6:
            return function[0](part.x)*function[1](part.px)*\
                   function[2](part.y)*function[3](part.py)*\
                   function[4](part.zeta)*function[5](part.delta)
        else:
            raise NotImplementedError


    def lossmap(self, part, interpolation=0.1, file=None, recompute=False, weights=None):

        # We cache the result
        if (self._lossmap is None or self._part is None
            or not xt.line._dicts_equal(part.to_dict(), self._part)
            or recompute
           ):

            self._part = part.to_dict()

            # Loss location refinement
            if interpolation is not None:
                # We need to correct for particles scattered out of the collimator beyond the aperture restriction:
                # TODO: this should be done in the scattering code!!!
                new_state = part.state
                new_elem  = part.at_element
                for idx, elem in enumerate(part.at_element):
                    if (part.state[idx]==0 and elem > 0 and
                        self.line.element_names[elem-1] in self.collimator_names
                       ):
                        print(f"Found at {self.line.element_names[elem]}, should be {self.line.element_names[elem-1]}")
                        new_state[idx] = -339 # lost on collimator, special state  ## TODO get the correct flag
                        new_elem[idx] = elem - 1
                part.state      = new_state
                part.at_element = new_elem
                # Do the interpolation
                aper_s = list(part.s[part.state==0])
                if len(aper_s) > 0:
                    print("Performing the aperture losses refinement.")
                    loss_loc_refinement = xt.LossLocationRefinement(self.line,
                            n_theta = 360, # Angular resolution in the polygonal approximation of the aperture
                            r_max = 0.5, # Maximum transverse aperture in m
                            dr = 50e-6, # Transverse loss refinement accuracy [m]
                            ds = interpolation, # Longitudinal loss refinement accuracy [m]
                            # save_refine_trackers=True # Diagnostics flag
                            )
                    loss_loc_refinement.refine_loss_location(part)

            aper_s, aper_names, aper_nabs = self._get_aperture_losses(part, weights)
            coll_summary = self.summary(part, weights, recompute=recompute, show_zeros=False).to_dict('list')

            self._lossmap = {
                'collimator': {
                    's':      coll_summary['s'],
                    'name':   coll_summary['collname'],
                    'length': coll_summary['length'],
                    'n':      coll_summary['nabs']
                }
                ,
                'aperture': {
                    's':    aper_s,
                    'name': aper_names,
                    'n':    aper_nabs
                }
                ,
                'machine_length': self.machine_length
                ,
                'interpolation': interpolation
                ,
                'reversed': self._line_is_reversed
            }

        if file is not None:
            with open(Path(file), 'w') as fid:
                json.dump(self._lossmap, fid, cls=xo.JEncoder, indent=True)
    
        return self._lossmap


    def _get_aperture_losses(self, part, weights=None):
        if weights is None:
            weights = np.ones(len(part.x))
        else:
            part.sort(interleave_lost_particles=True)

        # Get s position per particle (lost on aperture)
        aper_mask = part.state==0
        aper_s = list(part.s[aper_mask])
        if len(aper_s) == 0:
            return [], [], []
        if self._line_is_reversed:
            aper_s = [ self.machine_length - s for s in aper_s ]

        # Store names of aperture markers
        aper_names   = [self.line.element_names[i] for i in part.at_element[aper_mask]]
        name_dict    = dict(zip(aper_s, aper_names)) # TODO: not floating-point-safe and slow

        # Create output arrays
        aper_pos     = np.unique(aper_s)
        aper_weights = weights[aper_mask]
        aper_nabs    = [aper_weights[aper_s==ss].sum() for ss in aper_pos] # TODO: this might be slow
        aper_names   = [name_dict[ss] for ss in aper_pos]
        return aper_pos, aper_names, aper_nabs

