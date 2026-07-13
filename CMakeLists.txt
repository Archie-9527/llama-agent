cmake_minimum_required(VERSION 3.21)

# =============================================================================
# llama-agent — Top-Level Build System
# =============================================================================
# This project integrates three sub-projects:
#   1. llama-cpp-python  — C/C++ native libraries (llama.cpp) + Python bindings
#   2. langchain          — Pure Python (langchain-core, langchain, partners)
#   3. agent-wrapper      — Pure Python (llama_agent package)
#
# The native C/C++ build is delegated to llama-cpp-python/CMakeLists.txt.
# Pure Python sub-projects are managed via pip/uv install; this file provides
# convenience targets for development workflows (build, test, lint, install).
# =============================================================================

project(llama-agent
    VERSION 0.1.0
    DESCRIPTION "LangChain + llama-cpp-python agent integration"
    LANGUAGES C CXX
)

# ---------------------------------------------------------------------------
# 1. Platform detection & cross-platform settings
# ---------------------------------------------------------------------------

if(APPLE)
    set(LLAMA_AGENT_PLATFORM "macOS")
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE LLAMA_AGENT_HOST_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Platform: macOS (${LLAMA_AGENT_HOST_ARCH})")
elseif(WIN32)
    set(LLAMA_AGENT_PLATFORM "Windows")
    message(STATUS "Platform: Windows (${CMAKE_GENERATOR})")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LLAMA_AGENT_PLATFORM "Linux")
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE LLAMA_AGENT_HOST_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Platform: Linux (${LLAMA_AGENT_HOST_ARCH})")
else()
    message(WARNING "Unknown platform: ${CMAKE_SYSTEM_NAME}")
    set(LLAMA_AGENT_PLATFORM "Unknown")
endif()

# ---------------------------------------------------------------------------
# 2. Build options
# ---------------------------------------------------------------------------

option(LLAMA_AGENT_BUILD_NATIVE  "Build llama.cpp shared libraries"  ON)
option(LLAMA_AGENT_INSTALL_PYTHON "Install Python packages (pip install -e)" OFF)

# llama.cpp backend options — passed through to llama-cpp-python
option(LLAMA_AGENT_CUBLAS  "Enable CUDA backend"   OFF)
option(LLAMA_AGENT_METAL   "Enable Metal backend"   ON)
option(LLAMA_AGENT_VULKAN  "Enable Vulkan backend"  OFF)
option(LLAMA_AGENT_HIP     "Enable HIP/ROCm backend" OFF)

# Python toolchain
set(PYTHON_EXECUTABLE "" CACHE FILEPATH "Path to the Python interpreter (default: auto-detect)")

# ---------------------------------------------------------------------------
# 3. Python discovery
# ---------------------------------------------------------------------------

if(NOT PYTHON_EXECUTABLE)
    find_program(PYTHON_EXECUTABLE
        NAMES python3 python
        DOC "Python interpreter"
    )
endif()

if(NOT PYTHON_EXECUTABLE)
    message(WARNING "Python interpreter not found — Python install targets disabled")
else()
    message(STATUS "Python: ${PYTHON_EXECUTABLE}")
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} --version
        OUTPUT_VARIABLE LLAMA_AGENT_PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _py_ok
    )
    if(_py_ok EQUAL 0)
        message(STATUS "Python version: ${LLAMA_AGENT_PYTHON_VERSION}")
    endif()
endif()

# ---------------------------------------------------------------------------
# 4. Native library build (llama.cpp)
# ---------------------------------------------------------------------------

if(LLAMA_AGENT_BUILD_NATIVE)

    # Propagate backend options to the llama-cpp-python sub-build
    if(LLAMA_AGENT_CUBLAS)
        set(LLAMA_CUBLAS ON CACHE BOOL "llama.cpp: enable CUDA" FORCE)
    endif()
    if(LLAMA_AGENT_HIP)
        set(LLAMA_HIP ON CACHE BOOL "llama.cpp: enable HIP" FORCE)
    endif()
    if(APPLE AND LLAMA_AGENT_METAL)
        set(GGML_METAL ON CACHE BOOL "ggml: enable Metal" FORCE)
        set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "ggml: embed metal library" FORCE)
    endif()
    if(LLAMA_AGENT_VULKAN)
        set(GGML_VULKAN ON CACHE BOOL "ggml: enable Vulkan" FORCE)
    endif()

    # Include the llama-cpp-python build which handles:
    #   - vendor/llama.cpp   (llama, ggml, ggml-* backends)
    #   - vendor/llama.cpp/tools/mtmd  (llava/multi-modal)
    # Libraries are installed to llama_cpp/lib/
    #
    # We use add_subdirectory rather than ExternalProject because the sources
    # are vendored in-tree and we want the targets available directly.
    add_subdirectory(llama-cpp-python)

    message(STATUS "Native library targets registered: llama, ggml, ggml-base, mtmd")
endif()

# ---------------------------------------------------------------------------
# 5. Installation
# ---------------------------------------------------------------------------

# Collect Python packages to install in editable mode
set(LLAMA_AGENT_PYTHON_PACKAGES
    # llama-cpp-python  (native libs handled above; Python package next)
    llama-cpp-python
    # langchain core
    langchain/libs/core
    # langchain
    langchain/libs/langchain_v1
    # agent-wrapper
    agent-wrapper
)

# Collect langchain partner packages
set(LLAMA_AGENT_PARTNER_PACKAGES
    langchain/libs/partners/anthropic
    langchain/libs/partners/chroma
    langchain/libs/partners/deepseek
    langchain/libs/partners/exa
    langchain/libs/partners/fireworks
    langchain/libs/partners/groq
    langchain/libs/partners/huggingface
    langchain/libs/partners/mistralai
    langchain/libs/partners/nomic
    langchain/libs/partners/ollama
    langchain/libs/partners/openai
    langchain/libs/partners/openrouter
    langchain/libs/partners/perplexity
    langchain/libs/partners/qdrant
    langchain/libs/partners/xai
)

# ---- install-all (CMake native libs + pip install all Python packages) ----
if(PYTHON_EXECUTABLE)
    set(_pip_install_cmd ${PYTHON_EXECUTABLE} -m pip install -e --quiet)

    # Install all core Python packages in editable mode
    add_custom_target(pip-install-core
        COMMENT "Installing core Python packages in editable mode..."
        COMMAND ${_pip_install_cmd} ${CMAKE_SOURCE_DIR}/llama-cpp-python
        COMMAND ${_pip_install_cmd} ${CMAKE_SOURCE_DIR}/langchain/libs/core
        COMMAND ${_pip_install_cmd} ${CMAKE_SOURCE_DIR}/langchain/libs/langchain_v1
        COMMAND ${_pip_install_cmd} ${CMAKE_SOURCE_DIR}/agent-wrapper
    )

    # Install partner packages
    foreach(_partner ${LLAMA_AGENT_PARTNER_PACKAGES})
        if(EXISTS ${CMAKE_SOURCE_DIR}/${_partner}/pyproject.toml)
            add_custom_target(pip-install-${_partner}
                COMMAND ${_pip_install_cmd} ${CMAKE_SOURCE_DIR}/${_partner}
                COMMENT "Installing ${_partner}..."
            )
            add_dependencies(pip-install-partners pip-install-${_partner})
        endif()
    endforeach()

    # Aggregate target: install everything
    add_custom_target(install-all
        COMMENT "Installing all sub-projects..."
    )
    add_dependencies(install-all pip-install-core)
    if(TARGET pip-install-partners)
        add_dependencies(install-all pip-install-partners)
    endif()

    message(STATUS "Python install targets: pip-install-core, pip-install-partners, install-all")
endif()

# ---- install (standard CMake install) ----
# When the user runs "cmake --install .", install native libs if they were built
if(LLAMA_AGENT_BUILD_NATIVE)
    # llama-cpp-python subdirectory already defines its own install() rules
    # that place .so/.dylib/.dll files into llama_cpp/lib/
    # The top-level install() just ensures those are included.
    install(DIRECTORY ${CMAKE_BINARY_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        FILES_MATCHING
        PATTERN "*.so" PATTERN "*.so.*"
        PATTERN "*.dylib" PATTERN "*.dll"
        PATTERN "*.a"
    )
endif()

# ---------------------------------------------------------------------------
# 6. Testing
# ---------------------------------------------------------------------------

if(PYTHON_EXECUTABLE)
    # ---- native library smoke test ----
    add_test(
        NAME llama-cpp-test-import
        COMMAND ${PYTHON_EXECUTABLE} -c "import llama_cpp; print('llama_cpp OK:', llama_cpp.__version__)"
    )
    set_tests_properties(llama-cpp-test-import
        PROPERTIES LABELS "smoke"
    )

    # ---- agent-wrapper tests ----
    if(EXISTS ${CMAKE_SOURCE_DIR}/agent-wrapper/tests)
        add_test(
            NAME agent-wrapper-tests
            COMMAND ${PYTHON_EXECUTABLE} -m pytest
                ${CMAKE_SOURCE_DIR}/agent-wrapper/tests
                -v --tb=short
        )
        set_tests_properties(agent-wrapper-tests
            PROPERTIES LABELS "agent-wrapper;python"
        )
    endif()

    # ---- langchain-core tests ----
    if(EXISTS ${CMAKE_SOURCE_DIR}/langchain/libs/core/tests)
        add_test(
            NAME langchain-core-tests
            COMMAND ${PYTHON_EXECUTABLE} -m pytest
                ${CMAKE_SOURCE_DIR}/langchain/libs/core/tests/unit_tests
                -v --tb=short -n auto
        )
        set_tests_properties(langchain-core-tests
            PROPERTIES LABELS "langchain;python"
        )
    endif()

    # ---- CTest configuration ----
    enable_testing()
    message(STATUS "CTest targets registered for: smoke, agent-wrapper, langchain-core")
endif()

# ---------------------------------------------------------------------------
# 7. Linting & formatting (development helpers)
# ---------------------------------------------------------------------------

if(PYTHON_EXECUTABLE)
    # ---- ruff check ----
    add_custom_target(lint
        COMMENT "Running ruff linter across all Python sources..."
        COMMAND ${PYTHON_EXECUTABLE} -m ruff check
            ${CMAKE_SOURCE_DIR}/agent-wrapper/llama_agent
            ${CMAKE_SOURCE_DIR}/llama-cpp-python/llama_cpp
            ${CMAKE_SOURCE_DIR}/langchain/libs/core/langchain_core
            ${CMAKE_SOURCE_DIR}/langchain/libs/langchain_v1/langchain
    )

    # ---- ruff format check (dry-run) ----
    add_custom_target(format-check
        COMMENT "Checking Python formatting (ruff format --check)..."
        COMMAND ${PYTHON_EXECUTABLE} -m ruff format --check
            ${CMAKE_SOURCE_DIR}/agent-wrapper/llama_agent
            ${CMAKE_SOURCE_DIR}/llama-cpp-python/llama_cpp
            ${CMAKE_SOURCE_DIR}/langchain/libs/core/langchain_core
            ${CMAKE_SOURCE_DIR}/langchain/libs/langchain_v1/langchain
    )

    # ---- ruff format (apply) ----
    add_custom_target(format
        COMMENT "Formatting Python sources with ruff..."
        COMMAND ${PYTHON_EXECUTABLE} -m ruff format
            ${CMAKE_SOURCE_DIR}/agent-wrapper/llama_agent
            ${CMAKE_SOURCE_DIR}/llama-cpp-python/llama_cpp
            ${CMAKE_SOURCE_DIR}/langchain/libs/core/langchain_core
            ${CMAKE_SOURCE_DIR}/langchain/libs/langchain_v1/langchain
    )

    # ---- mypy type check ----
    add_custom_target(typecheck
        COMMENT "Running mypy type checks..."
        COMMAND ${PYTHON_EXECUTABLE} -m mypy
            ${CMAKE_SOURCE_DIR}/agent-wrapper/llama_agent
            --ignore-missing-imports
    )

    message(STATUS "Dev targets: lint, format-check, format, typecheck")
endif()

# ---------------------------------------------------------------------------
# 8. Clean
# ---------------------------------------------------------------------------

add_custom_target(clean-all
    COMMENT "Cleaning build artifacts..."
    # CMake build dir
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    # llama_cpp compiled libs
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_SOURCE_DIR}/llama-cpp-python/llama_cpp/lib
    # Python caches
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_SOURCE_DIR}/llama-cpp-python/llama_cpp/__pycache__
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_SOURCE_DIR}/agent-wrapper/llama_agent/__pycache__
)

# ---------------------------------------------------------------------------
# 9. Build summary (printed after configuration)
# ---------------------------------------------------------------------------

message(STATUS "")
message(STATUS "   ____  _                              __ ")
message(STATUS "  / __ )(_)___  ____ _   _____  _____  / /_")
message(STATUS " / __  / / __ \\/ __ \\ | / / _ \\/ ___/ / __/")
message(STATUS "/ /_/ / / / / / /_/ / |/ /  __/ /    / /_  ")
message(STATUS "/_____/_/_/ /_/\\____/|___/\\___/_/     \\__/  ")
message(STATUS "")
message(STATUS "  llama-agent build configuration")
message(STATUS "  =================================")
message(STATUS "  Platform:       ${LLAMA_AGENT_PLATFORM} (${LLAMA_AGENT_HOST_ARCH})")
message(STATUS "  Build native:   ${LLAMA_AGENT_BUILD_NATIVE}")
if(LLAMA_AGENT_BUILD_NATIVE)
    message(STATUS "  CUDA:           ${LLAMA_AGENT_CUBLAS}")
    message(STATUS "  Metal:          ${LLAMA_AGENT_METAL}")
    message(STATUS "  Vulkan:         ${LLAMA_AGENT_VULKAN}")
    message(STATUS "  HIP/ROCm:       ${LLAMA_AGENT_HIP}")
endif()
message(STATUS "  Python:         ${PYTHON_EXECUTABLE}")
message(STATUS "")
message(STATUS "  Targets:")
if(LLAMA_AGENT_BUILD_NATIVE)
    message(STATUS "    cmake --build .                  Build native libraries")
endif()
if(PYTHON_EXECUTABLE)
    message(STATUS "    cmake --build . --target pip-install-core")
    message(STATUS "                    Install core Python packages")
    message(STATUS "    cmake --build . --target install-all")
    message(STATUS "                    Install all Python packages")
    message(STATUS "    ctest                            Run all tests")
    message(STATUS "    cmake --build . --target lint    Run ruff linter")
    message(STATUS "    cmake --build . --target format  Auto-format Python sources")
    message(STATUS "    cmake --build . --target clean-all")
    message(STATUS "                    Remove all build artifacts")
endif()
message(STATUS "")
