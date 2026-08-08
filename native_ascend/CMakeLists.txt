cmake_minimum_required(VERSION 3.16)
project(agile_agent_ascend_contract LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(agile_agent_ascend_contract_stub SHARED
    src/contract_stub.cpp
)
target_include_directories(agile_agent_ascend_contract_stub PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(agile_agent_ascend_contract_stub PRIVATE
    AGILE_AGENT_ASCEND_BUILD
)
