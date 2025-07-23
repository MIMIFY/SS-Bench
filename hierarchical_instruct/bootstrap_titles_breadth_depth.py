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
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests

# Base prompt context for ASD social story title generation
base_prompt_context = "Imagine a Social Story book designed as an effective and meaningful approach to help children with Autism Spectrum Disorder (ASD) understand and navigate various social situations through stories. \
The ultimate and deeper goal is to empower children and older people by enhancing their understanding of social situations and social encounters in their lives, and thereby supporting their ability to be active participants in life’s routines and activities. \n\
The book contains chapters on different themes that are crucial for social interaction, emotional understanding and so on. \n"

# Prompt for generating story titles
base_title_expanding_prompt = "Please come up with or complete a series of potential Social Story titles given a chapter and its corresponding explanation. You must just give the Social Story Titles in the chapter without any other outputs.\n\n"

random.seed(42)


def encode_prompt(prompt_demonstration_tasks, request_task):
    """Encode multiple demonstrations and the request chapter into a single string."""
    prompt = base_prompt_context + base_title_expanding_prompt
    for idx, point in enumerate(prompt_demonstration_tasks):
        chapter = point['Chapter']
        titles = point['Social Story Titles in the chapter']
        explanation = point['Explanation']
        prompt += "Chapter: " + re.sub(r"\s+", " ", chapter).strip().rstrip(":") + "\n"
        prompt += "Chapter explanation: " + re.sub(r"\s+", " ", explanation).strip().rstrip(":") + "\n"
        prompt += "Social Story Titles in the chapter: \n"
        random.shuffle(titles)
        for h in range(len(titles[:15])):
            title = re.sub(r"\s+", " ", titles[h]).strip().rstrip(":")
            prompt += f"{h+1}. {title}\n"
        prompt += "\n"
    prompt += "Chapter: " + re.sub(r"\s+", " ", request_task['Chapter']).strip().rstrip(":") + "\n"
    prompt += "Chapter explanation: " + re.sub(r"\s+", " ", request_task['Explanation']).strip().rstrip(":") + "\n"
    prompt += "Social Story Titles in the chapter: \n"
    return prompt


def sample_machine_instructions(machine_tasks, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_tasks, min(n, len(machine_tasks)))


def find_word_in_string(w, s):
    """Find a word in a string using regex."""
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_gpt3_response(response):
    """Post-process GPT response to extract a list of story titles."""
    if response is None or response['choices'][0]['finish_reason'] == "length":
        print("Response is empty or stopped due to length limit:", response)
        return []
    print("Raw LLM response:", response['choices'][0]['message']['content'])
    # Split response by numbered items
    raw_titles = re.split(r"\n\d+\s?\. ", response['choices'][0]['message']['content'])
    raw_titles = [re.sub(r"^\d+\.\s", "", tit) for tit in raw_titles]
    print("After regex matching:", raw_titles)
    Titles = []
    for title in raw_titles:
        # Clean and normalize title
        title = re.sub(r"\s+", " ", title).strip()
        title = title.strip().capitalize()
        if title == "":
            print("Title is empty after processing:", title)
            continue
        # Filter by length (1-20 words)
        if len(title.split()) <= 0 or len(title.split()) > 20:
            print("Title length out of range:", len(title.split()), title)
            continue
        if title.startswith("This chapter has already"):
            print("Title starts with 'This chapter has already':", title)
            continue
        # Filter titles starting with punctuation
        punctuation = string.punctuation
        punctuation_without_asterisk = punctuation.replace('*', '')
        if title[0] in punctuation_without_asterisk:
            print("Title starts with punctuation:", title)
            continue
        # Filter non-ASCII characters
        if not title[0].isascii():
            print("Title starts with non-ASCII character:", title)
            continue
        Titles.append(title)
        print("Passed post-processing")
    return Titles


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
        # required=True,
        default="SS-GEN/seed_data_gen/seed_data/seed_chapter_title_list.jsonl",
        help="Path to human-written seed data."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_chapters_explanations.jsonl",
        help="Name of machine-generated chapters and explanations."
    )
    parser.add_argument(
        "--num_titles_to_generate",
        type=int,
        default=70,
        help="Number of titles to generate for each chapter."
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
        default=10,
        help="Number of examples to demonstrate in prompt."
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
    print(args.batch_dir)
    print(args.engine)
    print("******************************")
    # Load seed data from JSONL file
    with open(args.seed_stories_path,'r',encoding='utf-8') as fin:
        lines = fin.readlines()
        seed_tasks = []
        seed_new_tasks = []
        for line in lines:
            data  = json.loads(line)
            data1 = json.loads(line)
            seed_tasks.append(data)
            
            data1['Chapter_metadata'] = "original_story_book"
            data1['Social Story Titles in the chapter'] = []
            data1['Title_length'] = 0
            data1['Request_idx'] = 0
            data1 = OrderedDict(
                (k, data1[k]) for k in ["Chapter", "Explanation", "Social Story Titles in the chapter", "Title_length","Request_idx","Chapter_metadata"]
            )
            seed_new_tasks.append(data1)

    seed_titles_flat = [i for t in seed_tasks for i in t['Social Story Titles in the chapter']]
    seed_chapters = [t["Chapter"] for t in seed_new_tasks]
    seed_explanations = [t["Explanation"] for t in seed_new_tasks]

    print(f"Loaded {len(seed_chapters)} human-written seed chapters and explanations. --- seed tasks")
    # Load new chapters and explanations from input file
    with open(os.path.join(args.batch_dir, args.input_file)) as fin:
        lines = fin.readlines()
        new_tasks = []
        new_chapters = []
        new_explanations = []
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["Chapter_metadata"] = data["metadata"]
                del data["metadata"]
            if "chapter" in data:
                data["Chapter"] = data.pop('chapter')
            if "explanation" in data:
                data["Explanation"] = data.pop("explanation")
                
            data['Social Story Titles in the chapter'] = []
            data['Title_length'] = 0
            data['Request_idx'] = 0
            data = OrderedDict(
                (k, data[k]) for k in ["Chapter", "Explanation", "Social Story Titles in the chapter", "Title_length", "Request_idx","Chapter_metadata"]
            )
            new_chapters.append(data['Chapter'])
            new_explanations.append(data['Explanation'])
            new_tasks.append(data)
    print(f"Loaded {len(new_tasks)} LLM-augmented new chapters and explanations. ---- new tasks")

    all_tasks = new_tasks + seed_new_tasks
    all_chapters = new_chapters + seed_chapters
    all_explanations = new_explanations + seed_explanations
    assert len(all_chapters)==len(all_explanations)==len(all_tasks)

    progress_task_bar = tqdm.tqdm(total=len(all_tasks), desc="Outer loop for total chapters which need to be titles-expanded")
    # The progress bar iterates over all chapters (first new, then seed) for title generation

    os.makedirs(args.batch_dir, exist_ok=True)
    Title_generation_dir = os.path.join(args.batch_dir,"Titles generation")

    generated_titles_flat = []
    if os.path.exists(Title_generation_dir):
        # If there are existing generated files, load them to avoid duplication
        pattern = os.path.join(Title_generation_dir, '*.jsonl')
        jsonl_files = glob.glob(pattern)
        if len(jsonl_files) > 0:
            for file in jsonl_files:
                json_data = [json.loads(l) for l in open(file, "r")]
                if len(json_data) > 0:
                    chapter = json_data[0]['chapter']
                    titles = [i["title"] for i in json_data]
                    request_idx = [i["request_idx"] for i in json_data]
                    generated_titles_flat += titles
                    for task in all_tasks:
                        if task['Chapter'] == chapter:
                            task['Social Story Titles in the chapter'] = titles
                            task['Title_length'] = len(titles)
                            task['Request_idx'] = request_idx[-1] + 1
                            print("Updated Request_idx:",task['Request_idx'])
                            break
    os.makedirs(Title_generation_dir, exist_ok=True)

    for task_id in range(len(all_chapters)):
        if all_tasks[task_id]['Title_length']>=args.num_titles_to_generate:
            print("The ",task_id," th task, is already generated enough! Chapter name: ",all_chapters[task_id])
            progress_task_bar.update(1)
        else:
            assert all_tasks[task_id]['Chapter']==all_chapters[task_id]
            print("Current task id: ",task_id," current chapter name:",all_chapters[task_id])
            filename = clean_filename(all_chapters[task_id])
            task_outfile_path = os.path.join(Title_generation_dir, filename + ".jsonl")
            
            progress_title_bar = tqdm.tqdm(total=args.num_titles_to_generate,desc=f"Inner loop for titles generation of current {task_id} th task.")
            if all_tasks[task_id]['Title_length']>0:
                progress_title_bar.update(all_tasks[task_id]['Title_length'])
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            with open(task_outfile_path, "a") as fout:
                while all_tasks[task_id]['Title_length'] < args.num_titles_to_generate:
                    args.num_prompt_demonstrations = random.randint(7, 12)
                    prompt_demonstration_tasks = random.sample(seed_tasks, args.num_prompt_demonstrations)
                    random.shuffle(prompt_demonstration_tasks)
                    request_task = all_tasks[task_id]
                    prompt = encode_prompt(prompt_demonstration_tasks = prompt_demonstration_tasks,request_task = request_task)
                    results = make_gpt3_requests(
                        engine=args.engine,
                        prompts=prompt, 
                        max_tokens=1024,
                        temperature=0.7,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=2,
                        stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                        logprobs=True,
                        n=1,
                        best_of=1,
                        api_key=args.api_key,
                        base_url=args.base_url,
                    )
                    current_chapter = all_tasks[task_id]['Chapter']
                    current_explanation = all_tasks[task_id]['Explanation']
                    current_new_titles = []
                    all_metadata = []
                    for result in results:
                        new_titles_list = post_process_gpt3_response(result["response"])
                        print("Number of results after post-processing:",len(new_titles_list))
                        current_new_titles += new_titles_list
                        all_metadata += [result] * len(new_titles_list)
                    # filter out highly similar titles
                    for tit, metadata in zip(current_new_titles, all_metadata):
                        all_titles_flat = seed_titles_flat + generated_titles_flat
                        with Pool(4) as p:
                            rouge_scores = p.map(partial(scorer.score, tit), all_titles_flat)
                        rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                        if max(rouge_scores) > 0.7:
                            print("Filtered out due to high similarity:",tit)
                            continue
                        most_similar_titles = {
                                all_titles_flat[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                        }
                        
                        generated_titles_flat.append(tit)
                        all_tasks[task_id]['Social Story Titles in the chapter'].append(tit)
                        all_tasks[task_id]['Title_length'] += 1
                        
                        fout.write(json.dumps({
                            "title":tit,
                            "chapter": current_chapter,
                            "explanation": current_explanation,
                            "most_similar_title": most_similar_titles,
                            "avg_similarity_score_tit": float(np.mean(rouge_scores)),
                            "title_metadata": metadata,
                            "request_idx": all_tasks[task_id]["Request_idx"]
                        }) + "\n")
                        progress_title_bar.update(1)
                    all_tasks[task_id]["Request_idx"] += 1
            progress_task_bar.update(1)

