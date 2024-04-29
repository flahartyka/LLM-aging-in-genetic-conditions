#!/bin/bash
source myconda

mamba activate base

cd llama.cpp

set -e

MODEL=./llama-2-70b-chat/ggml-model-q4_0.gguf
MODEL_NAME=management_answers

# exec options
opts="--temp 0 -n 300" # additional flags
nl='
'
introduction="Provide a management plan for each scenario."

# file options
question_file=./management_q.txt
touch ./llama-2-70b-chat/results/$MODEL_NAME.txt
output_file=./llama-2-70b-chat/results/$MODEL_NAME.txt

counter=1

echo 'Running'
while IFS= read -r question
do
  exe_cmd="./main -c 2048- -s 0 -p "\"$introduction$nl$question\"" "$opts" -m ""\"$MODEL\""" >> ""\"$output_file\""
  echo $counter
  echo "Current Question: $question"
  eval "$exe_cmd"
  echo -e "\n------" >> $output_file
  counter=$((counter+1))
done < "$question_file"


