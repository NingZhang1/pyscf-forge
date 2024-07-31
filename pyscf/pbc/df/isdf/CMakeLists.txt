# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(isdf SHARED 
  pbc_isdf_samplek.c pbc_isdf_V.c pbc_isdf_auxbasis.c pbc_isdf_sparse.c 
  pbc_isdf_eri.c)

set_target_properties(isdf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(isdf cgto cint cvhf np_helper ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})

if(ENABLE_FFTW)
add_library(fft SHARED fft.c)
set_target_properties(fft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})
target_link_libraries(fft fftw3_threads fftw3 ${BLAS_LIBRARIES})
endif()
