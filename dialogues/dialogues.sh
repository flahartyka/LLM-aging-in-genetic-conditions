#!/bin/bash
source myconda

mamba activate base

cd llama.cpp

set -e

python generate_dialogues.py

