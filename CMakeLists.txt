cmake_minimum_required(VERSION 3.20)
project(example)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)
# Other modules you could enable
#igl_include(embree)
#igl_include(imgui)
#igl_include(opengl)
#igl_include(stb)
#igl_include(predicates)
#igl_include(xml)
#igl_include(copyleft cgal)
#igl_include(copyleft comiso)
#igl_include(copyleft core)
#igl_include(copyleft cork)
igl_include(copyleft tetgen)
#igl_include(restricted matlab)
#igl_include(restricted mosek)
#igl_include(restricted triangle)

add_subdirectory(deps/polyscope)
add_subdirectory(deps/geometry-central)

# === Gurobi setup ===

# Set the Gurobi path, either manually or via environment variable
# You can adjust these paths based on where Gurobi is installed on your system.
set(GUROBI_HOME "/Library/gurobi1200/macos_universal2")  # Change this to your Gurobi installation directory
set(GUROBI_INCLUDE_DIR "${GUROBI_HOME}/include")
set(GUROBI_LIB_DIR "${GUROBI_HOME}/lib")

# If Gurobi was not found, raise an error
if (NOT GUROBI_HOME)
    message(FATAL_ERROR "Gurobi installation not found. Please set GUROBI_HOME.")
else()
    message(STATUS "Gurobi home directory set as ${GUROBI_HOME}")
endif()

# Add Gurobi include and lib directories to the project
include_directories(${GUROBI_INCLUDE_DIR})
link_directories(${GUROBI_LIB_DIR})
message(STATUS "Gurobi include dir: ${GUROBI_INCLUDE_DIR}")
message(STATUS "Gurobi library dir: ${GUROBI_LIB_DIR}")

# # Add Gurobi to the list of libraries to link to your project
# list(APPEND GC_DEP_LIBS Gurobi)

# If Gurobi is not already defined as a target, you can add it as an imported library.
add_library(Gurobi STATIC IMPORTED)
set_target_properties(Gurobi PROPERTIES
  IMPORTED_LOCATION "${GUROBI_LIB_DIR}/libgurobi_c++.a"  # Update this for your platform
)

add_library(Gurobi_dylib STATIC IMPORTED)
set_target_properties(Gurobi_dylib PROPERTIES
  IMPORTED_LOCATION "${GUROBI_LIB_DIR}/libgurobi120.dylib"  # Update this for your platform
)

# === Done with Gurobi setup ===

# Add your project files
file(GLOB SRC_FILES *.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/args")

# Link igl (and the glfw module) to your project
target_link_libraries(${PROJECT_NAME} PUBLIC 
  igl::glfw
  ## Other modules you could link to
  # igl::embree
  # igl::imgui
  # igl::opengl
  # igl::stb
  # igl::predicates
  # igl::xml
  # igl_copyleft::cgal
  # igl_copyleft::comiso
  # igl_copyleft::core
  # igl_copyleft::cork
  igl_copyleft::tetgen
  # igl_restricted::matlab
  # igl_restricted::mosek
  # igl_restricted::triangle
  polyscope
  geometry-central
  Gurobi
  Gurobi_dylib
  )
