# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=1
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
sub_block_ratio=0.5
# You can change the model path to LLaDA-1.5 by setting model_path='GSAI-ML/LLaDA-1.5'


# baseline
python eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
--limit 50 


# prefix cache+parallel factor
python eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,factor=${factor},show_speed=True \
--limit 50 


# dynamic block length+parallel
python eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},use_cache=True,dynamic_block_length=True,factor=${factor},show_speed=True \
--limit 50 

# dynamic block length + sub_block_ratio +parallel
python eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},use_cache=True,dynamic_block_length=True,factor=${factor},sub_block_ratio=${sub_block_ratio},show_speed=True \
--limit 50 