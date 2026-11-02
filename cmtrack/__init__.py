"""column-marker-tracking: camera-shake removal and 3x11 checker-marker tracking for RC column tests.

Modules
-------
config   : YAML/JSON settings with defaults
frames   : frame sources (image sequence, .npy stack, video file)
zncc     : ZNCC template matching with sub-pixel (parabolic) peak refinement
layout   : 3x11 marker grid, marker IDs, quadrant (checker) templates
camera   : camera-motion estimation from fixed points (translation / similarity)
markers  : marker initialisation and frame-to-frame tracking with quality flags
scale    : px -> mm conversion (known rectangle or known marker spacing)
clean    : jump detection (running-median residual) and PCHIP gap filling
deform   : adjacent-marker spacing, member angle theta, axial deformation dv
plots    : figures (time histories, dv-theta, camera motion)
pipeline : the four processing steps used by scripts/01-04
synth    : synthetic test video with known ground truth (demo / tests)
"""

__version__ = "0.1.0"
