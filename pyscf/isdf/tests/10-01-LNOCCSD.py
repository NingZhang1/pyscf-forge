# backend to test #

import pyscf.isdf.BackEnd._config as config

config.disable_fftw()
config.backend("torch")
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

MAX = BACKEND._maximum
ABS = BACKEND._absolute
ToTENSOR = BACKEND._toTensor

import numpy as np
import numpy

from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.tools import super_cell
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.dft import multigrid

from lno.cc import LNOCCSD
from lno.tools import guess_frozen

# isdf util #

from pyscf.isdf.isdf_tools_Tsym import _kmesh_to_Kpoints, _1e_operator_gamma2k
from pyscf.isdf import isdf_tools_cell
from pyscf.isdf.isdf import ISDF
from pyscf.isdf.isdf_local import ISDF_Local
from pyscf.isdf.isdf_to_df import DF_ISDF, GDF_ISDF

# test #

cell = pbcgto.Cell()

# Molecule
boxlen = 12.4138
cell.a = numpy.array([[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])
cell.atom = [
    ["O", (12.235322, 1.376642, 10.869880)],
    ["O", (6.445390, 3.706940, 8.650794)],
    ["O", (0.085977, 2.181322, 8.276663)],
    ["O", (12.052554, 2.671366, 2.147199)],
    ["O", (12.250036, 4.190930, 12.092014)],
    ["O", (7.187422, 0.959062, 4.733469)],
    ["O", (8.346457, 7.210040, 4.667644)],
    ["O", (12.361546, 11.527875, 8.106887)],
    ["O", (3.299984, 4.440816, 9.193275)],
    ["O", (2.855829, 3.759909, 6.552815)],
    ["O", (1.392494, 6.362753, 0.586172)],
    ["O", (1.858645, 8.694013, 2.068738)],
    ["O", (3.770231, 12.094519, 8.652183)],
    ["O", (6.432508, 3.669828, 2.772418)],
    ["O", (1.998724, 1.820217, 4.876440)],
    ["O", (8.248581, 2.404730, 6.931303)],
    ["O", (5.753814, 3.360029, 12.461534)],
    ["O", (11.322212, 5.649239, 2.236798)],
    ["O", (4.277318, 2.113956, 10.590808)],
    ["O", (5.405015, 3.349247, 5.484702)],
    ["O", (6.493278, 11.869958, 0.684912)],
    ["O", (3.275250, 2.346576, 2.425241)],
    ["O", (7.981003, 6.352512, 7.507970)],
    ["O", (5.985990, 6.512854, 12.194648)],
    ["O", (10.636714, 11.856872, 12.209540)],
    ["O", (9.312283, 3.670384, 3.508594)],
    ["O", (1.106885, 5.830301, 6.638695)],
    ["O", (8.008007, 3.326363, 10.869818)],
    ["O", (12.403000, 9.687405, 11.761901)],
    ["O", (4.219782, 7.085315, 8.153470)],
    ["O", (3.781557, 8.203821, 11.563272)],
    ["O", (11.088898, 4.532081, 7.809475)],
    ["O", (10.387548, 8.408890, 1.017882)],
    ["O", (1.979016, 6.418091, 10.374159)],
    ["O", (4.660547, 0.549666, 5.617403)],
    ["O", (8.745880, 12.256257, 8.089383)],
    ["O", (2.662041, 10.489890, 0.092980)],
    ["O", (7.241661, 10.471815, 4.226946)],
    ["O", (2.276827, 0.276647, 10.810417)],
    ["O", (8.887733, 0.946877, 1.333885)],
    ["O", (1.943554, 8.088552, 7.567650)],
    ["O", (9.667942, 8.056759, 9.868847)],
    ["O", (10.905491, 8.339638, 6.484782)],
    ["O", (3.507733, 4.862402, 1.557439)],
    ["O", (8.010457, 8.642846, 12.055969)],
    ["O", (8.374446, 10.035932, 6.690309)],
    ["O", (5.635247, 6.076875, 5.563993)],
    ["O", (11.728434, 1.601906, 5.079475)],
    ["O", (9.771134, 9.814114, 3.548703)],
    ["O", (3.944355, 10.563450, 4.687536)],
    ["O", (0.890357, 6.382287, 4.065806)],
    ["O", (6.862447, 6.425182, 2.488202)],
    ["O", (3.813963, 6.595122, 3.762649)],
    ["O", (6.562448, 8.295463, 8.807182)],
    ["O", (9.809455, 0.143325, 3.886553)],
    ["O", (4.117074, 11.661225, 2.221679)],
    ["O", (5.295317, 8.735561, 2.763183)],
    ["O", (9.971999, 5.379339, 5.340378)],
    ["O", (12.254708, 8.643874, 3.957116)],
    ["O", (2.344274, 10.761274, 6.829162)],
    ["O", (7.013416, 0.643488, 10.518797)],
    ["O", (5.152349, 10.233624, 10.359388)],
    ["O", (11.184278, 5.884064, 10.298279)],
    ["O", (12.252335, 8.974142, 9.070831)],
    ["H", (12.415139, 2.233125, 11.257611)],
    ["H", (11.922476, 1.573799, 9.986994)],
    ["H", (5.608192, 3.371543, 8.971482)],
    ["H", (6.731226, 3.060851, 8.004962)],
    ["H", (-0.169205, 1.565594, 7.589645)],
    ["H", (-0.455440, 2.954771, 8.118939)],
    ["H", (12.125168, 2.826463, 1.205443)],
    ["H", (12.888828, 2.969761, 2.504745)],
    ["H", (11.553255, 4.386613, 11.465566)],
    ["H", (12.818281, 4.960808, 12.067151)],
    ["H", (7.049495, 1.772344, 4.247898)],
    ["H", (6.353019, 0.798145, 5.174047)],
    ["H", (7.781850, 7.384852, 5.420566)],
    ["H", (9.103203, 6.754017, 5.035898)],
    ["H", (12.771232, 11.788645, 8.931744)],
    ["H", (12.018035, 10.650652, 8.276334)],
    ["H", (3.557245, 3.792529, 9.848846)],
    ["H", (2.543844, 4.884102, 9.577958)],
    ["H", (2.320235, 4.521250, 6.329813)],
    ["H", (2.872128, 3.749963, 7.509824)],
    ["H", (1.209685, 7.121391, 1.140501)],
    ["H", (2.238885, 6.038801, 0.894245)],
    ["H", (2.763109, 8.856353, 2.336735)],
    ["H", (1.329379, 9.047369, 2.783755)],
    ["H", (4.315639, 11.533388, 9.203449)],
    ["H", (3.098742, 12.433043, 9.244412)],
    ["H", (5.987369, 3.448974, 3.590530)],
    ["H", (5.813096, 3.419344, 2.086985)],
    ["H", (1.057126, 1.675344, 4.969379)],
    ["H", (2.248496, 2.292119, 5.670892)],
    ["H", (8.508264, 1.653337, 7.464411)],
    ["H", (8.066015, 2.034597, 6.067646)],
    ["H", (5.197835, 2.915542, 11.821572)],
    ["H", (6.630900, 3.329981, 12.079371)],
    ["H", (10.788986, 6.436672, 2.127933)],
    ["H", (11.657923, 5.463602, 1.359832)],
    ["H", (3.544476, 1.634958, 10.977765)],
    ["H", (4.755770, 1.455054, 10.087655)],
    ["H", (4.465371, 3.375459, 5.665294)],
    ["H", (5.682663, 4.264430, 5.524498)],
    ["H", (6.174815, 11.778676, 1.582954)],
    ["H", (5.713640, 12.089924, 0.174999)],
    ["H", (3.476076, 1.498708, 2.028983)],
    ["H", (2.730229, 2.134295, 3.182949)],
    ["H", (7.119624, 5.936450, 7.474030)],
    ["H", (8.536492, 5.799405, 6.958665)],
    ["H", (5.909499, 5.717477, 11.667621)],
    ["H", (6.125402, 6.196758, 13.087330)],
    ["H", (11.203499, 12.513536, 11.804844)],
    ["H", (10.260930, 12.300153, 12.970145)],
    ["H", (9.985036, 3.927685, 2.878172)],
    ["H", (8.545584, 3.468329, 2.972331)],
    ["H", (1.399882, 6.620092, 7.093246)],
    ["H", (0.963561, 6.112523, 5.735345)],
    ["H", (8.067363, 3.674002, 9.979955)],
    ["H", (8.000737, 2.375959, 10.756190)],
    ["H", (11.821629, 10.402510, 12.020482)],
    ["H", (12.206854, 8.983242, 12.379892)],
    ["H", (3.461473, 7.606485, 7.889688)],
    ["H", (3.844478, 6.304711, 8.560946)],
    ["H", (3.179884, 7.585614, 11.148494)],
    ["H", (4.401957, 7.652030, 12.039573)],
    ["H", (11.573777, 5.053211, 7.169515)],
    ["H", (10.342076, 4.186083, 7.320831)],
    ["H", (10.065640, 8.919194, 1.760981)],
    ["H", (9.629585, 8.322499, 0.439729)],
    ["H", (1.396302, 6.546079, 9.625630)],
    ["H", (1.405516, 6.479759, 11.138049)],
    ["H", (4.024008, 1.232518, 5.405828)],
    ["H", (4.736858, 0.579881, 6.571077)],
    ["H", (9.452293, 12.313381, 8.732772)],
    ["H", (8.976559, 11.502788, 7.545965)],
    ["H", (1.834701, 10.012311, 0.153462)],
    ["H", (3.295197, 9.836403, -0.204175)],
    ["H", (7.056724, 11.401702, 4.095264)],
    ["H", (6.499038, 10.020287, 3.825865)],
    ["H", (1.365541, 0.487338, 11.013887)],
    ["H", (2.501591, -0.428131, 11.417871)],
    ["H", (8.644279, 1.812362, 1.005409)],
    ["H", (8.142674, 0.388030, 1.112955)],
    ["H", (1.272659, 8.365063, 8.191888)],
    ["H", (2.142485, 8.877768, 7.063867)],
    ["H", (8.961493, 7.826192, 9.265523)],
    ["H", (9.227102, 8.487654, 10.601118)],
    ["H", (10.150144, 7.758934, 6.392768)],
    ["H", (10.596082, 9.187988, 6.167290)],
    ["H", (3.463106, 4.096188, 2.129414)],
    ["H", (3.919461, 4.539801, 0.755791)],
    ["H", (7.418998, 9.394959, 12.028876)],
    ["H", (7.430413, 7.883095, 12.106546)],
    ["H", (7.972905, 10.220334, 5.841196)],
    ["H", (7.675111, 9.631498, 7.203725)],
    ["H", (5.332446, 6.381336, 6.419473)],
    ["H", (5.000025, 6.434186, 4.943466)],
    ["H", (11.575078, 2.271167, 4.412540)],
    ["H", (11.219802, 0.847030, 4.783357)],
    ["H", (8.865342, 9.721516, 3.843998)],
    ["H", (10.000732, 10.719285, 3.758898)],
    ["H", (3.186196, 10.476397, 5.265333)],
    ["H", (4.407331, 11.335128, 5.013723)],
    ["H", (0.558187, 7.255936, 3.859331)],
    ["H", (0.341672, 5.789383, 3.552346)],
    ["H", (7.459933, 6.526049, 3.229193)],
    ["H", (6.696228, 5.483739, 2.440372)],
    ["H", (3.864872, 6.313007, 2.849385)],
    ["H", (2.876419, 6.621201, 3.953862)],
    ["H", (5.631529, 8.079145, 8.753997)],
    ["H", (7.003296, 7.568245, 8.367822)],
    ["H", (9.615413, 0.527902, 3.031755)],
    ["H", (8.962985, 0.109366, 4.332162)],
    ["H", (3.825854, 11.139182, 1.474087)],
    ["H", (4.063988, 11.063232, 2.967211)],
    ["H", (5.784391, 7.914558, 2.708486)],
    ["H", (4.780461, 8.655167, 3.566110)],
    ["H", (10.880659, 5.444664, 5.046607)],
    ["H", (9.593331, 4.687991, 4.797350)],
    ["H", (11.562317, 8.960134, 3.376765)],
    ["H", (11.926084, 8.816948, 4.839320)],
    ["H", (2.856874, 11.297981, 7.433660)],
    ["H", (1.492332, 11.195517, 6.786033)],
    ["H", (7.145820, 0.090200, 9.749009)],
    ["H", (7.227275, 0.077690, 11.260665)],
    ["H", (4.662021, 9.538430, 10.798155)],
    ["H", (5.994537, 9.833472, 10.142985)],
    ["H", (10.544299, 6.595857, 10.301445)],
    ["H", (11.281750, 5.653082, 9.374494)],
    ["H", (12.103020, 8.841164, 10.006916)],
    ["H", (11.491592, 8.576221, 8.647557)],
]
cell.basis = "gth-tzv2p"
cell.ke_cutoff = 200  # kinetic energy cutoff in a.u.
cell.max_memory = 8000  # in MB
cell.precision = 1e-6  # integral precision
cell.pseudo = "gth-pade"
cell.verbose = 4
cell.use_loose_rcut = True  # integral screening based on shell radii
cell.use_particle_mesh_ewald = True  # use particle mesh ewald for nuclear repulsion
cell.build()

natm = cell.natm
group = [[i] for i in range(natm)]  # not the best way, but currently the only way!

### first search the parameter for ISDF ###

for qr_cutoff in [1e-3, 3e-4, 1e-4]:
    isdf = ISDF_Local(
        cell, with_robust_fitting=False, limited_memory=True, build_V_K_bunchsize=56
    )
    isdf.build(c=40, m=5, rela_cutoff=qr_cutoff, group=group)
    mf = scf.RHF(cell)
    mf.with_df = isdf
    mf.kernel()
