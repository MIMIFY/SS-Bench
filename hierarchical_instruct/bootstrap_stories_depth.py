import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
import glob
import sys
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from prompt_complete_social_story import encode_prompt

random.seed(42)

def post_process_gpt3_response(response):
    """Post-process GPT response to extract story content (introduction, main_body, conclusion)."""
    content_dict = {}
    if response is None or response['choices'][0]['finish_reason'] == "length":
        print("Response is empty or stopped due to length limit:", response)
        return content_dict
    response_text = response['choices'][0]['message']['content']
    print("Raw LLM response:", response_text)
    pattern = r'(\d*[\.,]?\s*#[^#]+#:)(\n(.*?)(?=\n\d*[\.,]?\s*#[^#]+#:|$))'
    # Find all matching sections for story parts
    matches = re.finditer(pattern, response_text, re.DOTALL)
    for match in matches:
        key, content_value, _ = match.groups()
        print("----")
        print(key, content_value)
        print("00000")
        pattern2 = r'\d*[\.,]?\s*#(.*?)#:'
        clean_key = re.sub(pattern2, r'\1', key).lower().replace(' ', '_')
        print("content_dict key:", clean_key)
        # Only accept expected keys
        assert_key = ["title", "introduction", "main_body", "conclusion"]
        if clean_key in assert_key:
            content_dict[clean_key] = content_value.strip()
        else:
            print("Unexpected key after post-processing:", clean_key)
            return content_dict
    # Validate keys and filter
    Flag = False
    if len(content_dict) == 4:
        specific_keys = ['title','introduction', 'main_body', 'conclusion']
        if all(key in specific_keys for key in content_dict):
            filtered_dict = {k: v for k, v in content_dict.items() if k in specific_keys[1:]}
            content_dict = filtered_dict
            Flag = True
        else:
            print("Key mismatch, post-processing failed")
            content_dict = {}
    elif len(content_dict) == 3:
        specific_keys = ['introduction', 'main_body', 'conclusion']
        if all(key in specific_keys for key in content_dict):
            Flag = True
        else:
            print("Key mismatch, post-processing failed")
            content_dict = {}
    else:
        print("Key mismatch, post-processing failed")
        content_dict = {}
    # Word count filtering
    words_number = 0
    if Flag:
        for key in content_dict:
            text = content_dict[key]
            if text == "":
                print("Empty text after key match, post-processing failed.")
                content_dict = {}
            else:
                words_number += len(text.split())
        if words_number < 25 or words_number > 564:
            print("Word count out of range:", words_number, "\nContent:", content_dict)
            content_dict = {}
        else:
            print("Passed strict post-processing!")
    return content_dict


def clean_filename(filename):
    """Clean filename by removing or replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    cleaned_filename = ''.join(char for char in filename if char not in invalid_chars)
    cleaned_filename = cleaned_filename.rstrip()
    cleaned_filename = ' '.join(cleaned_filename.split())
    if cleaned_filename.startswith('.'):
        cleaned_filename = '_' + cleaned_filename
    return cleaned_filename


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        # required=True,
        default="SS-GEN/hierarchical_instruct/data/gpt4_test_generations",
        help="Directory to store generated batches."
    )
    parser.add_argument(
        "--seed_stories_path",
        type=str,
        default="SS-GEN/seed_data_gen/seed_data/seed_chapter_story_list.jsonl",
        help="Path to human-written seed data."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="Titles generation",
    )
    parser.add_argument(
        "--output_seed_folder",
        type=str,
        default="Stories generation/Seed_Titles",
    )
    parser.add_argument(
        "--output_expand_folder",
        type=str,
        default="Stories generation/Generated_Titles_from_gpt4",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-4o",
        help="GPT model to use."
    )
    parser.add_argument(
        "--num_prompt_demonstrations",
        type=int,
        default=4,
        help="Number of instructions to use and be demonstrated in the prompt."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-xxxxxxxxxxxxxxxxxxx",
        help="OpenAI API key."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="xxxxxxxxxxxxxxxxxxx",
        help="OpenAI API base URL."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Ensure output directory exists
    if not os.path.exists(args.batch_dir):
        print(f"Error: The directory '{args.batch_dir}' does not exist. Exiting story generation.")
        sys.exit(1)
    # Part I: Generate stories for seed titles
    print('Part I: Start Social Story generation of Seed titles! ----- from (seed) chapter, chapter explanation.')
    # Load seed data from JSONL file
    with open(args.seed_stories_path,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        seed_tasks = []
        seed_tasks_for_regen = []
        for line in lines:
            data  = json.loads(line)
            seed_tasks.append(data)
            keys_to_keep = ['chapter', 'explanation', 'title']
            filtered_data = {k: v for k, v in data.items() if k in keys_to_keep}
            filtered_data['title_metadata'] = "original_story_book"
            filtered_data['request_idx'] = 0
            filtered_data = OrderedDict(
                (k, filtered_data[k]) for k in ["title", "chapter", "explanation", "title_metadata","request_idx"]
            )
            seed_tasks_for_regen.append(filtered_data)
    seed_chapters = [t["chapter"] for t in seed_tasks]
    print(f"Loaded {len(seed_tasks)} human-written seed chapters, seed stories, and LLM-augmented seed chapter explanations. --- seed tasks")
    # Load already generated stories for seed titles
    Story_generation_seed_dir = os.path.join(args.batch_dir, args.output_seed_folder)
    regen_already_done_seed_stories = []
    request_idx = 0
    if os.path.exists(os.path.join(Story_generation_seed_dir, "machine_generated_chapter_story_list.jsonl")):
        with open(os.path.join(Story_generation_seed_dir, "machine_generated_chapter_story_list.jsonl"), "r", encoding='utf-8') as fin:
            for line in fin:
                story_info = json.loads(line)
                regen_already_done_seed_stories.append(story_info)
                request_idx = story_info["request_idx"] + 1
        print(f"Loaded {len(regen_already_done_seed_stories)} LLM re-generated Social Stories of seed titles.")
    os.makedirs(Story_generation_seed_dir, exist_ok=True)
    regen_seed_stories_progress_bar = tqdm.tqdm(total=len(seed_tasks_for_regen),desc="Process of Social Story re-generation of Seed titles.")
    if regen_already_done_seed_stories:
        regen_seed_stories_progress_bar.update(len(regen_already_done_seed_stories))
    if len(regen_already_done_seed_stories)<len(seed_tasks_for_regen): 
        with open(os.path.join(Story_generation_seed_dir, "machine_generated_chapter_story_list.jsonl"), "a", encoding='utf-8') as fout:
            new_story_dict = {}
            for seed_regen in seed_tasks_for_regen[len(regen_already_done_seed_stories):]:
                while new_story_dict == {}:
                    prompt_demonstration_tasks = random.sample(seed_tasks, args.num_prompt_demonstrations)
                    random.shuffle(prompt_demonstration_tasks)
                    request_task = seed_regen
                    prompt = encode_prompt(prompt_demonstration_tasks = prompt_demonstration_tasks, request_task = seed_regen)
                    results = make_gpt3_requests(
                        engine=args.engine,
                        prompts=prompt, 
                        max_tokens=1024,
                        temperature=0.7,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=2,
                        # ["should","shouldn't","shouldn’t","supposed to","mustn’t","mustn't","ought","ought to know better","bad","naughty","never","always","can’t","can't","don’t","don't"]
                        stop_sequences=["Autistic","autistic", "autism", "autism", "you",  "You" ],      
                        logprobs=True,
                        n=1,
                        best_of=1,
                        api_key=args.api_key,
                        base_url=args.base_url,
                    )
                    result = results[0]
                    new_story_dict = post_process_gpt3_response(result["response"])
                    print(new_story_dict)
                    if new_story_dict:
                        current_title = seed_regen['title']
                        current_chapter = seed_regen['chapter']
                        current_explanation = seed_regen['explanation']
                        current_title_metadata = seed_regen['title_metadata']
                        story_metadata = [result] * 1
                        fout.write(json.dumps({
                            "title":current_title,
                            "chapter": current_chapter,
                            "explanation": current_explanation,
                            "introduction":new_story_dict['introduction'],
                            "main_body":new_story_dict['main_body'],
                            "conclusion":new_story_dict['conclusion'],
                            "story_metadata":story_metadata,
                            "title_metadata":current_title_metadata,
                            "request_idx": request_idx
                        }) + "\n")
                        regen_seed_stories_progress_bar.update(1)
                    request_idx += 1
                new_story_dict = {}   
    print("End Part I: Social Story generation of Seed titles!")

#=====================================================================================================================
    print('====================================\nPart II: Start Social Story generation of LLM-augmented titles! ----- from (LLM + seed) chapter, chapter explanation.')
    # Load generated titles for LLM-augmented chapters
    Title_generation_dir = os.path.join(args.batch_dir,args.input_folder)
    print("Loaded titles from:",Title_generation_dir)
    new_tasks_for_gen = []
    if not os.path.exists(Title_generation_dir):
        print(f"Error: The directory '{Title_generation_dir}' does not exist. Exiting story generation.")
        sys.exit(1)
    else:
        pattern = os.path.join(Title_generation_dir, '*.jsonl')
        jsonl_files = glob.glob(pattern)
        if len(jsonl_files) > 0:
            for file in jsonl_files:
                json_data = [json.loads(l) for l in open(file, "r",encoding='utf-8')]
                if len(json_data) > 0:
                    task_dict = {}
                    task_dict['chapter'] = json_data[0]['chapter']
                    task_dict['title_completion_length'] = 0
                    task_dict['titles_information'] = []
                    task_dict['request_idx'] = 0
                    for point in json_data:
                        keys_to_keep = ['title','chapter', 'explanation', 'title_metadata']
                        filtered_point = {k: v for k, v in point.items() if k in keys_to_keep}
                        filtered_point['story_flag'] = False
                        filtered_point = OrderedDict(
                            (k, filtered_point[k]) for k in ["title", "chapter", "explanation","story_flag", "title_metadata"]
                        )
                        task_dict['titles_information'].append(filtered_point)
                    new_tasks_for_gen.append(task_dict)
    print(f"Loaded {len(new_tasks_for_gen)} (LLM-augmented && Seed) chapters, along with explanations and story titles of each chapter. ---- new tasks")
    # Part II: Generate stories for LLM-augmented titles
    Story_generation_expand_dir = os.path.join(args.batch_dir, args.output_expand_folder)
    print("Stories will be saved to:",Story_generation_expand_dir)
    if os.path.exists(Story_generation_expand_dir):
        pattern = os.path.join(Story_generation_expand_dir, '*.jsonl')
        jsonl_files = glob.glob(pattern)
        if len(jsonl_files) > 0:
            for file in jsonl_files:
                json_data = [json.loads(l) for l in open(file, "r", encoding='utf-8')]
                if len(json_data) > 0:
                    chapter = json_data[0]['chapter']
                    titles = [i["title"] for i in json_data]
                    request_idx = [i["request_idx"] for i in json_data]
                    for task in new_tasks_for_gen:
                        if task['chapter'] == chapter:
                            task['title_completion_length'] = len(json_data)
                            for tit in task['titles_information']:
                                tit['story_flag'] = tit['title'] in titles  
                            task['request_idx'] = request_idx[-1] + 1
                            print("Updated request_idx:",task['request_idx'])
                            break
    os.makedirs(Story_generation_expand_dir, exist_ok=True)
    gen_new_tasks_progress_bar = tqdm.tqdm(total=len(new_tasks_for_gen),desc="Outer loop for total chapter tasks which need to be Story-Completed based on the Title.")
    count = 0
    for task_id in range(len(new_tasks_for_gen)):
        current_task_chapter = new_tasks_for_gen[task_id]['chapter']
        current_total_number = len(new_tasks_for_gen[task_id]['titles_information'])
        current_already_number = new_tasks_for_gen[task_id]['title_completion_length']
        if current_already_number == current_total_number:
            print("The ",task_id," th task, where the total stories are already generated ! Chapter name: ",current_task_chapter)
            gen_new_tasks_progress_bar.update(1)
        else:
            print("Current task id: ",task_id," current chapter name:",current_task_chapter)
            filename = clean_filename(current_task_chapter)
            task_outfile_path = os.path.join(Story_generation_expand_dir, filename +".jsonl")
            progress_story_bar = tqdm.tqdm(total=current_total_number,desc=f"Inner Loop for Social Stories completion of current {task_id} th task.")
            if current_already_number > 0:
                progress_story_bar.update(current_already_number)
            if current_already_number < current_total_number: 
                need_to_be_done = new_tasks_for_gen[task_id]['titles_information']
                for already in need_to_be_done[:current_already_number]:
                    assert already['story_flag'] == True
                with open(task_outfile_path, "a",encoding='utf-8') as fout:
                    new_story_dict = {}
                    for new_gen in need_to_be_done[current_already_number:]:
                        assert new_gen['story_flag'] == False
                        while new_story_dict == {}:
                            prompt_demonstration_tasks = random.sample(seed_tasks, args.num_prompt_demonstrations)
                            random.shuffle(prompt_demonstration_tasks)
                            request_task = new_gen
                            prompt = encode_prompt(prompt_demonstration_tasks = prompt_demonstration_tasks, request_task = request_task)
                            results = make_gpt3_requests(
                                engine=args.engine,
                                prompts=prompt, 
                                max_tokens=1024,
                                temperature=0.7,
                                top_p=0.5,
                                frequency_penalty=0,
                                presence_penalty=2,
                                # ["should","shouldn't","shouldn’t","supposed to","mustn’t","mustn't","ought","ought to know better","bad","naughty","never","always","can’t","can't","don’t","don't"]
                                stop_sequences=["Autistic","autistic", "autism", "autism", "you",  "You" ],  
                                logprobs=True,
                                n=1,
                                best_of=1,
                                api_key=args.api_key,
                                base_url=args.base_url,
                            )
                            result = results[0]
                            new_story_dict = post_process_gpt3_response(result["response"])
                            print(new_story_dict)
                            if new_story_dict:
                                current_title = new_gen['title']
                                current_chapter = new_gen['chapter']
                                current_explanation = new_gen['explanation']
                                story_metadata = [result] * 1
                                fout.write(json.dumps({
                                    "title":current_title,
                                    "chapter": current_chapter,
                                    "explanation": current_explanation,
                                    "introduction":new_story_dict['introduction'],
                                    "main_body":new_story_dict['main_body'],
                                    "conclusion":new_story_dict['conclusion'],
                                    "story_metadata":story_metadata,
                                    "request_idx": new_tasks_for_gen[task_id]["request_idx"]
                                }) + "\n")
                                progress_story_bar.update(1)
                            new_tasks_for_gen[task_id]["request_idx"] += 1
                        new_story_dict = {}
            gen_new_tasks_progress_bar.update(1)
        count += 1
    print(count)
    print("End Part II: Social Story generation of LLM-augmented titles!")



