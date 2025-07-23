import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from gpt3_api import make_requests as make_gpt3_requests

# Base prompt context for ASD social story generation
base_prompt_context = "Imagine a Social Story book designed as an effective and meaningful approach to help children with Autism Spectrum Disorder (ASD) understand and navigate various social situations through stories. \
The ultimate and deeper goal is to empower children and older people by enhancing their understanding of social situations and social encounters in their lives, and thereby supporting their ability to be active participants in life’s routines and activities. \n\
The book contains chapters on different themes that are crucial for social interaction, emotional understanding and so on. \n"

# Prompt for generating chapter explanations
base_chapter_explantion_prompt = "Please come up with a series of Chapters for this book, along with the Explanations about the chapter's focus, which highlight how the chapter supports ASD children in developing corresponding critical skills and understandings. \
Note: non-repeating and in the format of [chapter]:[explanation].\n"

random.seed(42)


def encode_prompt(prompt_chapter_explanation):
    """Encode multiple prompt chapter_explanation into a single string."""
    prompt = base_prompt_context + base_chapter_explantion_prompt

    for idx, chapter_explanation in enumerate(prompt_chapter_explanation):
        # Clean and format chapter_explanation
        chapter_explanation = re.sub(r"\s+", " ", chapter_explanation).strip().rstrip(":")
        prompt += f"{idx+1}. {chapter_explanation}\n"
    
    return prompt


def sample_machine_chapter_explanation(machine_chapter_explanation, n):
    """Sample n machine chapter_explanation from a list of machine chapter_explanation."""
    return random.sample(machine_chapter_explanation, min(n, len(machine_chapter_explanation)))


def find_word_in_string(w, s):
    """Find a word in a string using regex."""
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_gpt3_response(response):
    """Post-process GPT response to extract chapter and explanation pairs."""
    if response is None or response['choices'][0]['finish_reason'] == "length":
        print("Response is empty or stopped due to length limit:", response)
        return []
    
    print("Raw LLM response:", response['choices'][0]['message']['content'])
    
    # Split response by numbered items
    raw_chapter_explanation = re.split(r"\n\d+\s?\. ", response['choices'][0]['message']['content']) 
    print("After regex matching:", raw_chapter_explanation)
    
    chapter_explanation = []
    for inst in raw_chapter_explanation:
        # Check for required asterisk format
        if "*" not in inst:
            print("Invalid format - missing asterisk:", inst)
            continue
        
        # Clean and normalize chapter_explanation
        inst = re.sub(r"\s+", " ", inst).strip() 
        inst = inst.strip().capitalize()
        
        if inst == "":
            print("Empty chapter_explanation after processing:", inst)
            continue
        
        # Filter by length (3-150 words)
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            print("chapter_explanation length out of range:", len(inst.split()), inst)
            continue        
        
        # Filter chapter_explanation starting with punctuation
        punctuation = string.punctuation
        punctuation_without_asterisk = punctuation.replace('*', '')
        if inst[0] in punctuation_without_asterisk:
            print("chapter_explanation starts with punctuation:", inst)
            continue
        
        # Filter non-ASCII characters
        if not inst[0].isascii():
            print("chapter_explanation starts with non-ASCII character:", inst)
            continue
        
        # Extract chapter and explanation using regex
        match = re.search(r'(\*([^:]+?)\*):\s+(.+)', inst)
        if match:
            chapter = match.group(2).strip()
            explanation = match.group(3).strip()
            
            output = {
                "chapter": chapter,
                "explanation": explanation
            }
            chapter_explanation.append(output)
            print("Extracted:", output)
        else:
            print("Format does not match expected pattern:", inst)
            continue
        
        print("Passed post-processing")
    
    return chapter_explanation


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
        default="SS-GEN/seed_data_gen/seed_data/seed_chapters_explanations.jsonl",
        help="Path to human-written seed data."
    )
    parser.add_argument(
        "--num_chapters_to_generate",
        type=int,
        default=56,
        help="Number of chapter_explanation to generate."
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
        default=8, 
        help="Number of chapter_explanation to demonstrate in prompt."
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
    
    # Load seed data from JSONL file
    seed_stories = [json.loads(l) for l in open(args.seed_stories_path, "r")]
    
    # Extract seed chapter_explanation, chapters, and explanations
    seed_chapter_explanation = ["*"+t["chapter"] + "*: " + t['explanation'] for t in seed_stories]
    seed_chapters = [t["chapter"] for t in seed_stories]
    seed_explanations = [t["explanation"] for t in seed_stories]

    print(f"Loaded {len(seed_chapter_explanation)} human-written seed chapters and explanations.")
    
    # Create output directory
    os.makedirs(args.batch_dir, exist_ok=True)

    # Initialize request index
    request_idx = 0
    
    # Load existing machine-generated chapters
    machine_chapter_explanation = []
    machine_chapters = []
    machine_explanations = []
    
    existing_file = os.path.join(args.batch_dir, "machine_generated_chapters_explanations.jsonl")
    if os.path.exists(existing_file):
        with open(existing_file, "r") as fin:
            for line in fin:
                chapter_explanation_info = json.loads(line)
                machine_chapters.append(chapter_explanation_info['chapter'])
                machine_explanations.append(chapter_explanation_info['explanation'])
                machine_chapter_explanation.append("*"+chapter_explanation_info["chapter"] + "*: " + chapter_explanation_info['explanation'])
                request_idx = chapter_explanation_info["request_idx"] + 1
        print(f"Loaded {len(machine_chapter_explanation)} existing machine-generated chapters and explanations.")

    # Initialize Rouge scorer for similarity checking
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    # Generate new chapters along with their explanations
    progress_bar = tqdm.tqdm(total=args.num_chapters_to_generate)
    if machine_chapter_explanation:
        progress_bar.update(len(machine_chapter_explanation))

    with open(existing_file, "a") as fout:
        while len(machine_chapter_explanation) < args.num_chapters_to_generate:
            
            # Sample machine-generated chapter_explanation for prompt
            prompt_chapter_explanation = sample_machine_chapter_explanation(machine_chapter_explanation, n=4)
            
            # Add human seed examples to prompt
            args.num_prompt_demonstrations = random.randint(8, 14)
            prompt_chapter_explanation += random.sample(seed_chapter_explanation, args.num_prompt_demonstrations - len(prompt_chapter_explanation))
            random.shuffle(prompt_chapter_explanation)
            
            # Encode prompt and call API
            prompt = encode_prompt(prompt_chapter_explanation)
            results = make_gpt3_requests(
                engine=args.engine,
                prompts=prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=True,
                n=1,
                best_of=1,
                api_key=args.api_key,
                base_url=args.base_url,
            )
            
            # Process results
            chapter_explanation = []
            chapters = []
            explanations = []
            all_metadata = []
            
            for result in results:
                new_dicts_list = post_process_gpt3_response(result["response"])
                print("Number of results after post-processing:", len(new_dicts_list))
                
                new_chapters = [i['chapter'] for i in new_dicts_list]
                new_explanations = [i['explanation'] for i in new_dicts_list]
                new_chapter_explanation = ["*"+i["chapter"] + "*: " + i['explanation'] for i in new_dicts_list]
                
                chapter_explanation += new_chapter_explanation
                chapters += new_chapters
                explanations += new_explanations
                all_metadata += [result] * len(new_chapter_explanation)

            # Filter and save results
            for inst, chap, expl, metadata in zip(chapter_explanation, chapters, explanations, all_metadata):
                # Calculate Rouge scores for similarity checking
                with Pool(4) as p:
                    rouge_scores1 = p.map(partial(scorer.score, inst), seed_chapter_explanation + machine_chapter_explanation)
                    rouge_scores2 = p.map(partial(scorer.score, chap), seed_chapters + machine_chapters)
                    rouge_scores3 = p.map(partial(scorer.score, expl), seed_explanations + machine_explanations)
                
                # Extract F-measure scores
                rouge_scores1 = [score["rougeL"].fmeasure for score in rouge_scores1]
                rouge_scores2 = [score["rougeL"].fmeasure for score in rouge_scores2]
                rouge_scores3 = [score["rougeL"].fmeasure for score in rouge_scores3]
                
                # Filter out highly similar chapter_explanation (Rouge-L > 0.7)
                if max(rouge_scores1) > 0.7 or max(rouge_scores2) > 0.7 or max(rouge_scores3) > 0.7:
                    print("Filtered out due to high similarity:", inst)
                    continue
                
                # Get most similar chapter_explanation for reference
                all_chapter_explanation = seed_chapter_explanation + machine_chapter_explanation
                all_chapters = seed_chapters + machine_chapters
                all_explanations = seed_explanations + machine_explanations
                
                most_similar_chapter_explanation = {
                    all_chapter_explanation[i]: rouge_scores1[i] for i in np.argsort(rouge_scores1)[-10:][::-1]
                }
                most_similar_chapters = {
                    all_chapters[i]: rouge_scores2[i] for i in np.argsort(rouge_scores2)[-10:][::-1]
                }
                most_similar_explanations = {
                    all_explanations[i]: rouge_scores3[i] for i in np.argsort(rouge_scores3)[-10:][::-1]
                }
                
                # Add to machine chapter_explanation
                machine_chapter_explanation.append(inst)
                machine_chapters.append(chap)
                machine_explanations.append(expl)
                
                # Save to file
                fout.write(json.dumps({
                    "chapter": chap,
                    "explanation": expl,
                    "most_similar_inst": most_similar_chapter_explanation,
                    "avg_similarity_score_inst": float(np.mean(rouge_scores1)),
                    "most_similar_chap": most_similar_chapters,
                    "avg_similarity_score_chap": float(np.mean(rouge_scores2)),
                    "most_similar_expl": most_similar_explanations,
                    "avg_similarity_score_expl": float(np.mean(rouge_scores3)),
                    "metadata": metadata,
                    "request_idx": request_idx
                }) + "\n")
                
                progress_bar.update(1)
            
            request_idx += 1
