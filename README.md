### Prompt Engineering Improves Video Understanding
This repository contains the code for our paper, "Prompt Engineering Improves Video Understanding."

#### Directory Structure
./llama: 
Contains code utilizing LLaMA 3.1 for generating accuracy scores, rewriting original questions, and generating automatic prefixes.

./llava_next_video: 
Includes code for using the LLaVA-NeXT-Video-7B-DPO model to perform inference across all prompt settings.

./llava_one_vision: 
Contains code for using the LLaVA-OneVision-Qwen2-7B-OV-Chat model to perform inference across all prompt settings.

./qwen2_vl: 
Features code for using the Qwen2-VL-7B-Instruct model to perform inference across all prompt settings.

#### Scripts and Examples
./inference.sh:
 An example script demonstrating how to perform inference on the TGIF dataset using the LLaVA-NeXT-Video-7B-DPO model with all prompt settings.

./prompts.py: 
Contains all the prompts used in our experiments.

