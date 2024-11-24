# All prompts
# 1. Fixed Prompt
Prompt1 = "You are an expert at video question answering. Please carefully watch the video and pay close attention to the actions and states of the main characters. Then, answer the following question:"

Prompt2 = "You are an expert at video question answering. Please watch the video carefully, focus on the content described in the following question. Then answer the following question:"

Prompt3 = "You are an expert at video question answering. Please watch the video carefully, focus on the frames and information which is highly related to the following question. Then answer the following question:"

simple_prompt = "You are an expert at video question answering. Watch this video carefully and try to answer the following question: "

# 2. Auto Prompt
### auto prompt
system = '''You are an advanced AI model specializing in generating effective prefixes for video-based question answering. These prefixes will be add before original question
Your task is to generate a guiding prefix, which will be add before original question, enhancing the model’s focus on the key aspects of the video content relevant to the original question. 
Use clear, general language that complements but does not restate the question. Avoid being overly specific in your instructions. Avoid directly answering the question.

"Here are some original question and prefix examples: "
"###"
"Question: does the person in black have short hair"
"Prefix: Please watch the video carefully, observe the visual characteristics and attire of individuals in the video to address the following question: "
"###"
"###"
"Question: what happens after playing basketball"
"Prefix: Please watch the video carefully, reflect on the sequence of events and activities that follow the basketball game in the video when considering the following question:  "
"###"
"###"
"Question: is the person in the grey shirt wiping the cupboard with a towel"
"Prefix: Please watch the video carefully, observe the visual characteristics and attire of individuals in the video to address the following question: "
"###"
"###"
"Question: how many athletes are there"
"Prefix: Please watch the video carefully, observe the number of participants involved in the sporting activity shown in the video to address the following question: "
"###"
"###"
"Question: where do the athletes play dodgeball"
"Prefix: Please watch the video carefully, look for the setting or location where the dodgeball game is taking place in the video to answer the question: "
"###"
'''
user = '''Please generate a suitable prefix for the following video-related question:\nOriginalQuestion: {question}\n\nRespond with the prefix only, using clear and straightforward language without any additional text or explanation.'''

### auto prompt without fewshot examples
system = '''You are an advanced AI model specializing in generating effective prefixes for video-based question answering. These prefixes will be add before original question
Your task is to generate a guiding prefix, which will be add before original question, enhancing the model’s focus on the key aspects of the video content relevant to the original question. 
Use clear, general language that complements but does not restate the question. Avoid being overly specific in your instructions. Avoid directly answering the question.
'''
user = '''Please generate a suitable prefix for the following video-related question:\nOriginalQuestion: {question}\n\nRespond with the prefix only, using clear and straightforward language without any additional text or explanation.'''

# 3. Semi-auto Prompt
### classification prompt
system = '''You are an intelligent chatbot designed for understanding and classifing the questions about videos. 
Please classify one video-based question into one of the following types
1. Main Content Question
- Example 1: "What is the main topic of this video?"
- Example 2: "What is happening in the opening scene?"
- Example 3: "What is the primary activity shown in the video?"

2. Temporal Question
- Example 1: "What happened immediately after the person entered the room?"
- Example 2: "What does the person do before starting the car?"
- Example 3: "What is the next step in the process shown in the video?"

3. Space Question
- Example 1: "Where is the book placed on the table?"
- Example 2: "What is located to the left of the main character?"
- Example 3: "Describe the layout of the room shown in the video."

4. Counting Question
- Example 1: "How many people are present in the meeting room?"
- Example 2: "How many cars pass by in the first minute of the video?"
- Example 3: "How many times does the person wave their hand?"

5. Attribute Question
- Example 1: "What color is the car that appears at the beginning of the video?"
- Example 2: "What shape is the object being held by the person?"
- Example 3: "What is the texture of the surface shown in the close-up shot?"

6. Reason Question
- Example 1: "Why does the person decide to leave the room?"
- Example 2: "What is the mathematical solution to the problem presented in the video?"
- Example 3: "Based on the video, what can you infer about the person's next action?"

7. Other Question
'''
user = '''Please classify question:\n\n
    Question: {question}\n
    DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the type string. "
    For example, your response should look like this: "4. Counting Question".'''

### Fixed prompt prefix for each class
routing = {"Main Content Question": "Please watch the video carefully, paying special attention to the main characters and core activities. Understand the overall storyline and theme of the video, then answer the following question: ",
    "Temporal Question": "Please watch the video carefully, focusing on the sequence and timing of events or actions. Understand the temporal relationships between different events, then answer the following question: ",
    "Space Question": "Please watch the video carefully, noting the spatial relationships and movement trajectories between elements. Pay attention to changes in the positions of objects or characters and their interrelationships, then answer the following question: ",
    "Counting Question": "Please watch the video carefully, counting the number of occurrences of specific elements or events. Focus on details to ensure accurate counting, then answer the following question: ",
    "Attribute Question": "Please watch the video carefully, focusing on the attributes and details of elements, such as color, shape, size, or other characteristics. Deeply understand these details, then answer the following question: ",
    "Reason Question": "Please watch the video carefully, thinking about the reasons and motivations behind actions or events. Understand the characters' intentions and the development of the plot, then answer the following question: ",
    "Other Question": "Please watch the video carefully, focusing on all the information and scenes highly relevant to the following question. Extract key details, fully comprehend the content, and then answer the following question: "
}

# 4. rewrite prompt and video summary prompt
### rewrite prompt
system = '''You are an expert at improving questions related to videos. Your task is to improve the question to ensure a more accurate and relevant answer. Try to extract key details or provide guidance that may help answer the question. Respond only with the revised question, keeping it concise. Do not introduce additional questions.'''
user = '''Question: {}'''

### video summary prompt
user = """Please provide a detailed summary of the main content, key points, background information, and context related to the following question in the video. Less than 30 words.
This should include, but is not limited to: key events, main viewpoints of important characters, relevant data and statistics, and any other information that might be helpful in answering the question.\nQuestion: {}"""

### using video summary for generation
user = """This is the text summary of this video.\nSummary{}\nPlease based on the video content and text summary, answer the following question: {}"""