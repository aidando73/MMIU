import os
import json
import base64
from openai import OpenAI
from PIL import Image
import io

# Configure API client
base_url = os.getenv('OPENAI_API_BASE', 'https://api.fireworks.ai/inference/v1/chat/completions')
api_key = os.getenv('FIREWORKS_API_KEY', None)

# If base_url doesn't include /v1/chat/completions, add it
if base_url and not base_url.endswith('/v1/chat/completions') and not base_url.endswith('/v1/chat/completions/'):
    if base_url.endswith('/v1'):
        base_url = base_url + '/chat/completions'
    elif not base_url.endswith('/'):
        base_url = base_url + '/v1/chat/completions'
    else:
        base_url = base_url + 'v1/chat/completions'

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

def encode_image(image_path):
    """Encode image to base64."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            
            return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def call_qwen3_vl(image_paths, question, model_name="qwen3-vl"):
    """Call qwen3-vl via OpenAI-compatible API."""
    content = []
    
    # Add images
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return 'image error'
        
        base64_image = encode_image(image_path)
        if base64_image is None:
            return 'image error'
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    # Add text question
    content.append({
        "type": "text",
        "text": question
    })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        print(f"ASSISTANT: {answer}")
        return answer
    except Exception as e:
        print(f"Model error: {e}")
        return 'model error'

json_path = 'all.json'

tasks_exist = ['person_reid', 'multiple_image_captioning', 'spot_the_similarity', 'face_retrieval', 'sketch2image_retrieval', 'handwritten_retrieval', 'spot_the_diff', 'image2image_retrieval', 'vehicle_retrieval', 'text2image_retrieval',
'general_action_recognition', 'video_captioning', 'next_img_prediction', 'temporal_ordering', 'meme_vedio_understanding', 'action_quality_assessment', 'temporal_localization', 'mevis',
'ravens_progressive_matrices', 'threed_indoor_recognition', 'point_tracking', 'threed_cad_recognition', 'single_object_tracking']

# Model name - can be overridden via environment variable
model_name = os.getenv('QWEN3_VL_MODEL', 'qwen3-vl')

if not os.path.exists(json_path):
    print(f"Error: {json_path} not found!")
    print("Please create all.json with the MMIU dataset.")
    exit(1)

with open(json_path, 'r') as f:
    data_all = json.load(f)

# Organize results by task
results_by_task = {}

for task_data in data_all:
    context = task_data["context"]
    question = task_data["question"]
    
    tmp = []
    image_flag = True
    
    for image_path in task_data["input_image_path"]:
        tmp.append(image_path)
        if not os.path.exists(image_path):
            image_flag = False
            break
    
    if image_flag == False:
        response = 'image none'
        task_data[model_name] = response
        print(f"{model_name}, {task_data.get('task', 'unknown')}, {len(tmp)}: {response}")
        task_name = task_data.get('task', 'unknown')
        if task_name not in results_by_task:
            results_by_task[task_name] = []
        results_by_task[task_name].append(task_data)
        continue
    
    try:
        if task_data['task'] in tasks_exist:
            question_formatted = question + '\n' + context
        else:
            question_formatted = context + '\n' + question
        question_formatted = question_formatted + '\nPlease answer the option directly like A,B,C,D...'
        
        response = call_qwen3_vl(tmp, question_formatted, model_name=model_name)
        task_data[model_name] = response
        print(f"{model_name}, {task_data.get('task', 'unknown')}, {len(tmp)}: {response}")
    except Exception as e:
        response = 'model error or image error'
        task_data[model_name] = response
        print(f"{model_name}, {task_data.get('task', 'unknown')}, {len(tmp)}: {response}")
        print(f"Exception: {e}")
    
    task_name = task_data.get('task', 'unknown')
    if task_name not in results_by_task:
        results_by_task[task_name] = []
    results_by_task[task_name].append(task_data)

# Save results organized by task
base_output_dir = os.path.join('../results')
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

for task_name, task_results in results_by_task.items():
    task_dir = os.path.join(base_output_dir, task_name)
    model_dir = os.path.join(task_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    output_path = os.path.join(model_dir, 'metadata_info.json')
    with open(output_path, 'w') as f:
        json.dump(task_results, f)
    print(f"Saved {len(task_results)} results for task '{task_name}' to {output_path}")

print(f"\nAll results saved to: {base_output_dir}")

