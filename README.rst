|Travis| |Codecov|

*****************************************
VarConLib - The Varying Constants Library
*****************************************

Daniel Berke

A library of functions and scripts for dealing with HARPS observations
in pursuit of my PhD.

Within the ``varconlib`` folder, there are:

* ``config``: contains a config file for defining the paths of various
  frequently-used directories.

* ``conversion``: library model containing functions for converting between air
  and vacuum wavelengths.

* ``data``: various files containing data used by other functions.

* ``exceptions``: custom exceptions.

* ``fitting``: library module containing code relevant to fitting models.

* ``miscellaneous``: library module containing various bits of random code that
  don't fit elsewhere, such as utility function to convert between wavelengths
  and velocities.

* ``obs1d``: library module containing code for opening 1D HARPS FITS files.
  (Old, deprecated.)

* ``obs2d``: library module containing code for opening 2D HARPS FITS files from
  an intermediate stage of the reduction pipeline.

* ``scripts``: scripts for data analysis (Kind of a mess.)

* ``star``: library module for the ``Star`` class which handles collating all
  the observational data for a star (which can have an arbitrary number of
  observations) and extracting the pair separation measurements necessary for
  measuring a change in alpha.

* ``transition_line``: library module containing the ``Transition`` class which
  holds code relating to a single atomic transition.

* ``transition_pair``: library module containing the ``TransitionPair`` class
  which holds two ``Transition`` objects as well as additional meta-data about
  them.



.. |Travis| image:: https://travis-ci.com/DBerke/varconlib.svg?branch=master
    :alt: Travis Badge
    :target: https://travis-ci.com/DBerke/varconlib

.. |Codecov| image:: https://codecov.io/gh/DBerke/varconlib/branch/master/graph/badge.svg
    :alt: Codecov Badge
    :target: https://codecov.io/gh/DBerke/varconlib