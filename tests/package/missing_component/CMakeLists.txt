cmake_minimum_required(VERSION 3.25)
project(microLLMMissingComponent LANGUAGES CXX)

find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS component_that_does_not_exist)
