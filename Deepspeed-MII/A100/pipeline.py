import argparse
import mii
import random
import string
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--max_input_length", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

def generate_random_word(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

def generate_input(args):
    random_words = ["France" for _ in range(args.max_input_length)]

    input_id = ""

    for word in random_words:
        input_id = input_id + word + " "

    input_id = input_id[:-1]
    
    input_list = []

    for batch_size in range(args.batch_size):
        input_list.append(input_id)
        
    return input_list

pipe = mii.pipeline(args.model)

prompts = generate_input(args)
#warmup 
responses = pipe(prompts, max_new_tokens=args.max_new_tokens, return_full_text=True)
#prefile
start_time_prefile = time.time()
responses = pipe(prompts, max_new_tokens=1, return_full_text=True)
end_time_prefile = time.time()
#decoding
start_time_decode = time.time()
responses = pipe(prompts, max_new_tokens=args.max_new_tokens, return_full_text=True)
end_time_decode = time.time()

latency_prefile = end_time_prefile-start_time_prefile
latency_decode = end_time_decode-start_time_decode-latency_prefile
throughput_prefile = (args.batch_size*(args.max_input_length + 1))/latency_prefile
throughput_decode = (args.batch_size*(args.max_new_tokens-1))/latency_decode


file_path = "llama_bench_results.csv"

if pipe.is_rank_0:
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        data = [["Nvidia A100 GPU",str(args.num_gpus),"Deepspeed-MII",args.model,str(args.max_input_length),str(args.batch_size),str(latency_prefile),str(throughput_prefile),str(latency_decode),str(throughput_decode)]]
        writer.writerows(data)


if pipe.is_rank_0:
    for r in responses:
        print(r, "\n", "-" * 80, "\n")

pipe.destroy()
