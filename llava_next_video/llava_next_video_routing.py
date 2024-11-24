import os
import json
import math
import argparse
import warnings
import traceback
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

ROUTING = {"Main Content Question": "Please watch the video carefully, paying special attention to the main characters and core activities. Understand the overall storyline and theme of the video, then answer the following question: ",
    "Temporal Question": "Please watch the video carefully, focusing on the sequence and timing of events or actions. Understand the temporal relationships between different events, then answer the following question: ",
    "Space Question": "Please watch the video carefully, noting the spatial relationships and movement trajectories between elements. Pay attention to changes in the positions of objects or characters and their interrelationships, then answer the following question: ",
    "Counting Question": "Please watch the video carefully, counting the number of occurrences of specific elements or events. Focus on details to ensure accurate counting, then answer the following question: ",
    "Attribute Question": "Please watch the video carefully, focusing on the attributes and details of elements, such as color, shape, size, or other characteristics. Deeply understand these details, then answer the following question: ",
    "Reason Question": "Please watch the video carefully, thinking about the reasons and motivations behind actions or events. Understand the characters' intentions and the development of the plot, then answer the following question: ",
    "Other Question": "Please watch the video carefully, focusing on all the information and scenes highly relevant to the following question. Extract key details, fully comprehend the content, and then answer the following question: "
}

def get_prompt(question:str):
    prompt = f"USER: <video>\n{question} ASSISTANT:"
    return prompt


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class VideochatgptDataset(Dataset):

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
        prefix_type = sample['prefix_type']
        prefix = ROUTING.get(prefix_type,"Please watch the video carefully, focusing on all the information and scenes highly relevant to the following question. Extract key details, fully comprehend the content, and then answer the following question: ")
        question = prefix + sample['question']
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
        try:
            video = VideoAsset(name=video_path,
                           num_frames=args.num_frames).np_ndarrays
        except Exception as e:
            return None
        
        return {
            'video':       video,
            'video_name':  video_name,
            'question':    question,
            'answer':      answer,
        }


def collate_fn(batch):
    vid  = [x['video'] for x in batch if x is not None]
    v_id = [x['video_name'] for x in batch if x is not None]
    qus  = [x['question'] for x in batch if x is not None]
    #qid  = [x['question_id'] for x in batch]
    ans  = [x['answer'] for x in batch if x is not None]
    return vid, v_id, qus, ans


def run_inference(args):

    model = args.model_type
    modality = args.modality


    llm = LLM(model=args.model_path)
    stop_token_ids = None


    # Initialize the model
    # model, processor, tokenizer = model_init(args.model_path)

    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.answer_file, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    args.batch_size = 1
    dataset = VideochatgptDataset(gt_questions, gt_answers)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=64, stop_token_ids=stop_token_ids)

    inputs = []
    metadata = []
    # Iterate over each sample in the ground truth file
    for i, (videos, video_names, questions, answers) in enumerate(tqdm(dataloader)):
        for i in range(len(videos)):
            video = videos[i]
            video_name   = video_names[i]
            question     = questions[i]
            question = args.prefix + question
            answer       = answers[i]
    
            prompt = get_prompt(question)
    
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    modality: video
                },
            })
            metadata.append({
                #"question_id": question_id,
                "question": question,
                "answer": answer
            })
    
    print("outputs begin")
    print("outputs begin")
    print("outputs begin")
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    print("outputs finished")
    print("outputs finished")
    print("outputs finished")
    for k, o in enumerate(outputs):
        generated_text = o.outputs[0].text
        print("generated_text", generated_text)
        sample_set = {
            #'id': metadata[k]['question_id'],
            'question': metadata[k]['question'],
            'answer': metadata[k]['answer'],
            'pred': generated_text
        }
        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=64)
    parser.add_argument("--num-workers", type=int, required=False, default=16)
    parser.add_argument("--model-type", type=str, required=False, default='llava-next-video')
    parser.add_argument("--modality", type=str, required=False, default='video')
    parser.add_argument("--num-frames", type=int, required=False, default=8)


    args = parser.parse_args()

    run_inference(args)
