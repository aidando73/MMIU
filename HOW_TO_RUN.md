# How to Run MMIU Evaluation

This guide explains how to run the MMIU (Multimodal Multi-image Understanding) evaluation.

## Overview

The evaluation process consists of 3 main steps:

1. **Model Inference** - Generate predictions using vision-language models
2. **Answer Matching** - Convert free-form answers to multiple choice options (A, B, C, D, etc.)
3. **Accuracy Calculation** - Calculate accuracy scores from matched answers

## Prerequisites

1. **Download the MMIU Dataset**
   - Dataset available at: https://huggingface.co/datasets/FanqingM/MMIU-Benchmark
   - You need to download the dataset and prepare it in the expected format

2. **Install Dependencies**
   - Install VLMEvalKit dependencies (see VLMEvalKit directory)
   - Required packages: `openai`, `pandas`, `transformers` (version depends on model)
   - For different models, different transformers versions are needed:
     - transformers == 33.0: XComposer2, XComposer2_1.8b, qwen_base, idefics_9b_instruct, qwen_chat, flamingov2
     - transformers == 37.0: deepseek_vl_1.3b, deepseek_vl_7b
     - transformers == 40.0: idefics2_8b

3. **Prepare Data Format**
   - Create `all.json` file in the `VLMEvalKit/` directory
   - Each entry should have:
     ```json
     {
       "context": "instruction/context text",
       "question": "question text",
       "input_image_path": ["path/to/image1.jpg", "path/to/image2.jpg", ...],
       "task": "task_name",
       "options": "A. option1 B. option2 C. option3 D. option4",
       "output": "correct_answer"  // e.g., "a" or "A"
     }
     ```

## Step 1: Run Model Inference

### Option A: Using VLMEvalKit (Recommended for supported models)

Navigate to the `VLMEvalKit/` directory and run:

```bash
cd VLMEvalKit
python test_models.py
```

**Before running**, edit `test_models.py`:
- Set `models` list to the models you want to evaluate (line 17)
- Ensure `json_path = 'all.json'` points to your dataset file (line 18)
- Make sure image paths in `all.json` are correct and accessible

**Supported models in test_models.py:**
- XComposer2, XComposer2_1.8b, qwen_base, idefics_9b_instruct, qwen_chat, flamingov2

### Option B: Using Model-Specific Scripts

For specific models, use dedicated scripts:
- `test_Mantis.py` - For Mantis model
- `test_interlvl1.5.py` - For InternVL1.5 model
- `test_internvl2-pro.py` - For InternVL2-pro model

**Example for Mantis:**
```bash
cd VLMEvalKit
python test_Mantis.py
```

**Output:** 
- Results saved to `./result/{modelname}/metadata_info.json`
- Each entry includes the model's prediction in a new field named after the model

## Step 2: Convert Predictions to Multiple Choice

Run `evaluate.py` to match free-form answers to multiple choice options using GPT-4o-mini.

**Before running**, edit `evaluate.py`:
1. Set OpenAI API credentials (lines 12-15 and 36-41):
   ```python
   client = OpenAI(
       base_url='your_openai_api_base_url',
       api_key='your_openai_api_key',
   )
   ```

2. Set the model names to evaluate (line 102):
   ```python
   modelnames = ['model1', 'model2', ...]
   ```

3. Set the result directories (line 103):
   ```python
   directorys = ['path/to/result/directory']
   ```
   The script expects directory structure: `{directory}/{taskname}/{modelname}/metadata_info.json`

**Run:**
```bash
cd /home/aidan/fireworks/MMIU
python evaluate.py
```

**Output:**
- Creates `metadata_info_choice.json` in each `{directory}/{taskname}/{modelname}/` directory
- Adds a `{modelname}_choice` field with the matched option (A, B, C, D, or Z for no match)

## Step 3: Calculate Accuracy

Run `evaluate_correct.py` to calculate accuracy scores.

**Before running**, edit `evaluate_correct.py`:
1. Set the result directories (line 5-7):
   ```python
   directorys = ['path/to/result/directory']
   ```

2. Set model names (line 16):
   ```python
   modelnames = ['model1', 'model2', ...]
   ```

**Run:**
```bash
cd /home/aidan/fireworks/MMIU
python evaluate_correct.py
```

**Output:**
- Creates `Accuracy_data_all.csv` with accuracy scores per task and model
- Includes an "Overall" row with average accuracy across all tasks

## Directory Structure Expected

After Step 1, your directory structure should look like:

```
result/
├── task1/
│   ├── model1/
│   │   └── metadata_info.json
│   └── model2/
│       └── metadata_info.json
├── task2/
│   ├── model1/
│   │   └── metadata_info.json
│   └── model2/
│       └── metadata_info.json
...
```

After Step 2, each model directory will also have:
- `metadata_info_choice.json`

## Notes

- **API Costs**: Step 2 uses GPT-4o-mini API calls, which incurs costs
- **Parallel Processing**: `evaluate.py` uses multiprocessing (10 processes by default, line 138)
- **Error Handling**: Scripts handle missing images, model errors, and API errors gracefully
- **Task-Specific Formatting**: Some tasks have different question/context ordering (see `tasks_exist` list in test scripts)

## Troubleshooting

1. **"all.json not found"**: Make sure the dataset file is in the correct location
2. **"image none" errors**: Check that image paths in `all.json` are correct and files exist
3. **"GPT error"**: Check OpenAI API credentials and network connectivity
4. **Model import errors**: Ensure correct transformers version is installed for your model

## Example Workflow

```bash
# 1. Download and prepare MMIU dataset as all.json
# 2. Run inference
cd VLMEvalKit
python test_models.py

# 3. Convert to choices (edit evaluate.py first with API keys)
cd ..
python evaluate.py

# 4. Calculate accuracy (edit evaluate_correct.py first with directories)
python evaluate_correct.py

# 5. View results
cat Accuracy_data_all.csv
```

