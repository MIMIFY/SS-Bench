
# SS-GEN: A Social Story Generation Framework with Large Language Models (AAAI 2025 Oral)

[![Paper](https://img.shields.io/badge/Paper-AAAI%202025%20Oral-green)](https://ojs.aaai.org/index.php/AAAI/article/view/32119)
[![License: OpenRAIL](https://img.shields.io/badge/License-OpenRAIL-red)](https://huggingface.co/spaces/BigScience/OpenRAIL)
[![Dataset](https://img.shields.io/badge/Dataset-Available-blue)](https://huggingface.co/datasets/FMiMiY/SS-GEN)


## 🧠 Introduction

**SS-GEN** is a novel framework for **automated generation of Social Stories™** aimed at helping children with Autism Spectrum Disorder (ASD) better understand and navigate social situations. Social Stories have traditionally been written by psychologists under strict guidelines — however, this process is costly, time-consuming, and lacks diversity.

SS-GEN leverages **Large Language Models (LLMs)** and a custom-designed, constraint-driven strategy (STARSOW) to generate personalized, high-quality Social Stories at scale.

<p align="center">
    <img src="assets/intro-SS.png" alt="SS-GEN Overview" width="480"/>
</p>

## 📝 Abstract

> Children with Autism Spectrum Disorder (ASD) often struggle to interpret social cues and engage in daily routines. Social Stories™, designed to improve these abilities, are typically handcrafted by experts, limiting their scalability. To address this, we propose **SS-GEN**, a framework that prompts LLMs to generate constraint-compliant Social Stories using a novel strategy named **STARSOW**. We further curate a high-quality dataset via human filtering and propose a structured evaluation framework. Finally, we fine-tune lightweight open-source models on our dataset, achieving strong results with lower cost and easier deployment. SS-GEN represents a significant step in creating **accessible, affordable, and automated tools** to assist ASD communities.



## 🌳 Framework: STARSOW

The **STARSOW** pipeline includes:
1. **Taking Root**: Generating diverse chapter themes from seed examples.
2. **Branching Out**: Producing multiple Social Story titles under each chapter.
3. **Bearing Star Fruits**: Completing full stories from titles, guided by strict structural and narrative constraints.
4. **Gardening Work**: Rigorous filtering to ensure quality, relevance, and safety.

<p align="center">
    <img src="assets/data-generation.png" alt="SS-GEN Framework" width="480"/>

</p>

## 🛠️ Implementation: StarSOW Method

This repository provides a complete implementation of the StarSOW method for generating Social Stories datasets. The implementation is organized into two main components:

### 📁 Repository Structure

```
📂 SS-GEN/
├── seed_data_gen/              # Seed dataset generation
│   ├── seed_data/              # Generated seed data
│   ├── seed_ main.py           # Main script for seed data generation
│   ├── openai_access.py        # OpenAI API interface
│   ├── utils.py                # Utility functions
│   └── explain_chapter.py      # Chapter explanation generation
├── hierarchical_instruct/      # StarSOW pipeline implementation
│   ├── bootstrap_chapters_breadth.py      # Step 1: Chapter generation
│   ├── bootstrap_titles_breadth_depth.py  # Step 2: Title generation
│   ├── bootstrap_stories_depth.py         # Step 3: Story generation
│   ├── gpt3_api.py                        # GPT API wrapper
│   ├── prompt_complete_social_story.py    # Story generation prompts
│   ├── data/                              # Generated data and processing
│   ├── visualization/                     # Data visualization tools
│   └── command_generate_story.sh          # Batch generation script
├── SS-GEN Dataset/             # Final dataset
├── README.md
└── Technical Appendix.pdf      # Detailed prompt templates
```

### 🚀 Quick Start

#### Prerequisites

```bash
pip install openai tqdm numpy pandas rouge-score datasets
```

#### Step 1: Generate Seed Dataset

First, prepare your seed Social Stories data and generate chapter explanations:

```bash
cd seed_data_gen
python seed_ main.py
```

This will:
- Load your original Social Stories data (`pure-ori.json`)
- Generate chapter explanations using GPT
- Create structured seed data for the StarSOW pipeline

#### Step 2: Generate Chapters (Taking Root)

Expand the seed chapters to create a diverse set of chapter themes:

```bash
cd hierarchical_instruct
python bootstrap_chapters_breadth.py \
    --seed_stories_path ../seed_data_gen/seed_data/seed_chapters_explanations.jsonl \
    --batch_dir data/gpt4_test_generations \
    --num_chapters_to_generate 56 \
    --engine gpt-4o \
    --api_key YOUR_API_KEY \
    --base_url YOUR_BASE_URL
```

#### Step 3: Generate Titles (Branching Out)

For each chapter, generate multiple Social Story titles:

```bash
python bootstrap_titles_breadth_depth.py \
    --seed_stories_path ../seed_data_gen/seed_data/seed_chapter_title_list.jsonl \
    --input_file machine_generated_chapters_explanations.jsonl \
    --batch_dir data/gpt4_test_generations \
    --num_titles_to_generate 70 \
    --engine gpt-4o \
    --api_key YOUR_API_KEY \
    --base_url YOUR_BASE_URL
```

#### Step 4: Generate Stories (Bearing Star Fruits)

Complete the Social Stories from titles using strict structural constraints:

```bash
python bootstrap_stories_depth.py \
    --seed_stories_path ../seed_data_gen/seed_data/seed_chapter_story_list.jsonl \
    --input_folder Titles generation \
    --batch_dir data/gpt4_test_generations \
    --output_seed_folder Stories generation/Seed_Titles \
    --output_expand_folder Stories generation/Generated_Titles_from_gpt4 \
    --engine gpt-4o \
    --api_key YOUR_API_KEY \
    --base_url YOUR_BASE_URL
```

#### Step 4.1: Batch Generation (Optional)

Use the provided shell script for automated batch processing:

```bash
bash command_generate_story.sh
```

### 🔧 Configuration

#### API Configuration

Update your API credentials in the respective scripts:

```python
# In gpt3_api.py or individual scripts
api_key = "your-openai-api-key"
base_url = "your-api-base-url"  # Optional, for custom endpoints
```

#### Generation Parameters

Key parameters you can adjust:

- `--num_chapters_to_generate`: Number of chapters to generate (default: 56)
- `--num_titles_to_generate`: Titles per chapter (default: 70)
- `--num_prompt_demonstrations`: Few-shot examples in prompts (default: 4-10)
- `--engine`: LLM model (gpt-4o, gpt-3.5-turbo, etc.)

### 📊 Data Processing and Visualization

The `hierarchical_instruct/data/` folder contains:
- Generated chapters, titles, and stories
- Processing scripts for data cleaning


The `hierarchical_instruct/visualization/` folder includes:
- Data distribution analysis

### 🎯 Quality Control

The implementation includes several quality control mechanisms:

1. **Rouge-based Similarity Filtering**: Removes highly similar generated content
2. **Length Constraints**: Ensures stories meet minimum/maximum word counts
3. **Structural Validation**: Verifies proper story structure (introduction, main body, conclusion)
4. **Content Safety**: Filters inappropriate or unsafe content
5. **Human Refined**: Human-curated and further refined for the final SS-GEN dataset.

### 📈 Monitoring Progress

Each generation step provides progress bars and detailed logging:

```bash
# Example output
Loaded 14 human-written seed chapters and explanations.
Loaded 0 existing machine-generated chapters and explanations.
Outer loop for total chapters which need to be titles-expanded: 100%|██████████| 14/14 [02:15<00:00]
```

### 🔄 Resuming Generation

The system supports resuming interrupted generation:
- Automatically detects existing generated files
- Continues from the last completed step
- Maintains consistency with `request_idx` tracking


## 📊 Dataset

We construct a large-scale Social Story dataset:

| Item | Description |
|------|-------------|
| Chapters | 57 diverse themes |
| Titles in each chapter | >=70 |
| Total stories | 5,085 |
| Avg. chapter length (in words) | 2.46 |
| Avg. title length (in words) | 5.28 |
| Avg. story content length (in words) | 281.65 |
| Structure | Title + Introduction + Body + Conclusion |
| Constraints | Structural Clarity, Descriptive Orientation, Situational Safety |


### 🧪 Load Dataset via Hugging Face 🤗 (**Recommend**)

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("FMiMiY/SS-GEN")
```

🔗 [View on Hugging Face](https://huggingface.co/datasets/FMiMiY/SS-GEN)


## 📈 Results

- We fine-tuned several 2B–8B models (e.g., Gemma, Mistral, LLaMA3).
- Fine-tuned models significantly outperformed zero-shot baselines across BLEU, ROUGE, and BERTScore.
- Human evaluation confirmed improvements in **empathy**, **coherence**, and **narrative safety**.



## Citation

If you use SS-GEN or our dataset, please cite:

```bibtex
@inproceedings{feng2025ss,
  title={SS-GEN: A Social Story Generation Framework with Large Language Models},
  author={Feng, Yi and Song, Mingyang and Wang, Jiaqi and Chen, Zhuang and Bi, Guanqun and Huang, Minlie and Jing, Liping and Yu, Jian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={2},
  pages={1300--1308},
  year={2025}
}
```



## 🤝 Acknowledgments

This work is supported by Beijing Jiaotong University, Tsinghua University, and Tencent. Special thanks to psychologists, educators, and collaborators who helped shape and evaluate this project.



## 📬 Contact

If you have any questions, suggestions or feedback, feel free to  submmit a issue or contact : 

**Yi Feng** – yifeng@bjtu.edu.cn
