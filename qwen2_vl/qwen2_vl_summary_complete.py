import os
import json
import math
import argparse
import warnings
import traceback
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Qwen2-VL
def run_qwen2_vl():
    llm = LLM(
        model=args.model_path,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    return llm, processor



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class ActivitynetDataset(Dataset):

    video_formats = ['.mp4', '.webm', '.avi', '.mov', '.mkv']

    def __init__(self, questions, answers):
        self.questions = questions
        self.answers   = answers

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]
        answer = self.answers[idx]

        video_name  = sample['video_name']
        question    = sample['question']
        #question_id = sample['question_id']
        answer      = answer['answer']

        video_path = None
        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(args.video_folder, f"v_{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break
            # BUG: compatibility for MSVD, MSRVTT, TGIF
            temp_path = os.path.join(args.video_folder, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break
   #     print("video_path", video_path)

        # video = VideoAsset(name=video_path,
        #                    num_frames=args.num_frames).np_ndarrays

        return {
            'video_path':       video_path,
            'video_name':  video_name,
            'question':    question,
            'answer':      answer,
        }


def collate_fn(batch):
    vid  = [x['video_path'] for x in batch]
    v_id = [x['video_name'] for x in batch]
    qus  = [x['question'] for x in batch]
    #qid  = [x['question_id'] for x in batch]
    ans  = [x['answer'] for x in batch]
    return vid, v_id, qus, ans


def run_inference(args):


    llm, processor = run_qwen2_vl()


    # Initialize the model
    # model, processor, tokenizer = model_init(args.model_path)

    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.answer_file, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    args.batch_size = 1
    dataset = ActivitynetDataset(gt_questions, gt_answers)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    sampling_params = SamplingParams(temperature=0.0,top_p=0.001, repetition_penalty=1.05, max_tokens=256,stop_token_ids=[])

    inputs = []
    metadata = []
    video_inputs_list = []
    # Iterate over each sample in the ground truth file
    for i, (video_paths, video_names, questions, answers) in enumerate(tqdm(dataloader)):
        video_path = video_paths[0]
        video_name   = video_names[0]
        question     = questions[0]
        answer       = answers[0]

        summary = """Please provide a detailed summary of the main content, key points, background information, and context related to the following question in the video. 
This should include, but is not limited to: key events, main viewpoints of important characters, relevant data and statistics, and any other information that might be helpful in answering the question. Less than 30 words.\nQuestion: {}""".format(question)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": 224 * 224,
                        "max_pixels": 448 * 448,
                        "nframes": args.num_frames,
                    },
                    {"type": "text", "text": summary},
                ],
            },
        ]

 
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        try:
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception as e:
            print(str(e))
            continue
        
        mm_data = {}
        mm_data["video"] = video_inputs

        video_inputs_list.append(video_inputs)

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
        })
        metadata.append({
            "video_name": video_name,
            "question": question,
            "answer": answer
        }) 
    print("summarys begin")
    print("summarys begin")
    print("summarys begin")
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    print("summarys finished")
    print("summarys finished")
    print("summarys finished")
    
    for k, o in enumerate(outputs):
        generated_text = o.outputs[0].text
        metadata[k]['summary'] = generated_text

    del outputs
    
    # re-answer
    inputs = []

    for i, (video_paths, video_names, questions, answers) in enumerate(tqdm(dataloader)):
        video_path = video_paths[0]
        video_name   = video_names[0]
        question     = questions[0]
        answer       = answers[0]

        summary = metadata[i]['summary']

        question = """This is the text summary of this video.\nSummary{}\nPlease based on the video content and text summary, answer the following question: {}""".format(summary, question)
        
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": 224 * 224,
                        "max_pixels": 448 * 448,
                        "nframes": args.num_frames,
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]

 
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        mm_data = {}
        mm_data["video"] = video_inputs_list[i]

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
        })
    
    print("output begin")
    print("output begin")
    print("output begin")
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    print("output begin")
    print("output begin")
    print("output begin")

    for k, o in enumerate(outputs):
        generated_text = o.outputs[0].text
        #print("generated_text", generated_text)
        sample_set = {
            'video_name':  metadata[k]['video_name'],
            'question': metadata[k]['question'],
            'pred': generated_text,
            'answer': metadata[k]['answer'],
            'summary': metadata[k]['summary']
        }
        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()

    print("successfully saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=32)
    parser.add_argument("--model-type", type=str, required=False, default='qwen_vl_7b')
    parser.add_argument("--modality", type=str, required=False, default='video')
    parser.add_argument("--num-frames", type=int, required=False, default=8)


    args = parser.parse_args()

    run_inference(args)
