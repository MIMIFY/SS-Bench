# "I want you act as a Prompt Creator.\r\n\
# Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
# This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
# The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
# The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
# '#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"
import random
gpt_role = """I want you to act as a Social Story writer for SLPs who takes care of children with Autism Spectrum Disorder (ASD). \
You are now writing a Social Story book to provide ASD children socially meaningful information that is accurate and presented \
in a positive and reassuring manner. The book contains different Chapters and each Chapter has different specific Social Stories. \
The stories must be brief, clear, and directly relevant to the intended audience, particularly individuals with autism.\n\n"""

task_description = """Given the information about the [Book Chapter] where the Social Story to be generated is situated, \
and given a [Social Story] #title# under the given [Book Chapter], your task is to complete the [Social Story] \
with only three parts #Introduction#, #Main body#, #Conclusion#.\n\n"""

social_story_creating_criteria = """Your Social Story Completion process should follow below Criteria:\n\n\
Criterion 1 (Defined Process and Goal): The Social Story should accurately describes a context, skill or achievement \
(e.g. dinner time, bath time, waiting in line) to help the ASD child better understand what is happening during a specific situation \
that they might find confusing and/or distressing. It must in a format that is safe, descriptive, and meaningful. \
Avoid any self-deprecating or negative references about the audience.

Criterion 2 (Audience Understanding and Topic Identification): Utilize relevant information to improve understanding of the audience \
in relation to the situation, skill, or concept. Identify the specific topic and focus on the most critical information. \
Celebrate achievements where possible.

Criterion 3 (Structure and Format): Each story should have a clear #Title#, an #Introduction# that identifies the topic in a positive way, \
a detailed #Main body# describing the issue of situation that has been identified as the focus of the story, and a #Conclusion# that \
reinforces the main information and trying to end on a positive note. Tailor the format to the individual abilities, attention span, \
learning style, and interests of the audience.
 
Criterion 4 (Voice, Tone and Vocabulary): Maintain a positive and patient tone, and ensure the language is literally accurate and clear, \
make the ASD child audience to feel safe and accepted. Be cautious not to misrepresent the audience's experiences. \
Never blame the bad behaviours, but say it in the positive way, eg: "I will try to wait in line. My mom and dad will be proud of me."
 
Criterion 5 (Guiding Questions): The story should try to answer relevant 'WH' questions (Where, When, Who, What, How, Why) to provide context.

Criterion 6 (Literal Accuracy and Verb precision): The story should provide literally accurate information and use verbs accurately. \
Never guess the ASD audience reaction (eg: never write that "I will have fun from it" because the guess may not be accurate and be hurtful.)

Criterion 7 (Ensure Descriptiveness): The Story should ensure that the number of descriptive sentences is more than twice \
the number of coaching sentences, the more the better. Descriptive sentences are sentences that describe the facts relating to \
the situation clearly and objectively, whilst coaching sentences are sentences that describe or suggest responses or actions. \
While ensuring descriptiveness, this story can incorporate a mix of descriptive, perspective, coaching, self-coaching, team-coaching, \
affirmative, and partial sentences.

Criterion 8 (Narrative Perspective): The Story must write as the First and/or third person perspective. Never use the second to \
avoid aggression. If you need to explain negative behaviours you can only use the third person (eg, "Sometimes children find it difficult \
to share their toys.")\n\n"""



example_start_generation = """                                                                                                                                                                                                                                                                                                                               """

example_template = """[Book Chapter]: Chapter "{}" which "{}."\n\
[Social Story]:\n\
1. #Title#:\n\
{}
2. #Introduction#:\n\
{}
3. #Main body#:\n\
{}
4. #Conclusion#:\n\
{}\n\n"""

reinforce_note_tips = """Note: The total word count of the generated #Introduction#, #Main body#, #Conclusion# must not exceed 400 words. \
Try to repeat the #Title# in the #Introduction# and #Conclusion# to reinforce the main message. \
Additionally, Social Story often uses parallelism to make itself catchy, and key information can be emphasized through \
the use of parallel structures and a call-and-response approach. \
Feel free to conceive and pen down a vivid and engaging backstory, delving into the experiences you imagine for the audience \
from their past, present, or future, which enriches the context when elucidating a concept or item. And It's best to include past tense, \
present tense, and future tense in the Story, for instance: "Last year, the people brought wrapped gifts. Hunter and I are hoping they \
will do that again! I told Grandpa Hill that I wish they wouldn't wrap the gifts. He says many people like to wrap gifts, so his guess \
is that they will wrap them this year, too." \n\n"""

generation_template = """[Book Chapter]: Chapter "{}" which "{}"\n\
[Social Story]:\n\
1. #Title#:\n\
{}
2. """                                                                                                                                                         
random.seed(42)

def encode_prompt(prompt_demonstration_tasks, request_task):
    """Encode multiple prompt instructions into a single string."""
    
    prompt = gpt_role + task_description + social_story_creating_criteria + example_start_generation
    random.shuffle(prompt_demonstration_tasks)
    for idx, point in enumerate(prompt_demonstration_tasks):
        chapter = point['chapter']
        explanation = point['explanation']
        title = point['title']
        introduction = point['introduction']
        main_body = point['main_body']
        conclusion = point['conclusion']
        prompt += example_template.format(chapter,explanation,title,introduction,main_body,conclusion)
    prompt += reinforce_note_tips
    prompt += generation_template.format(request_task['chapter'],request_task['explanation'],request_task['title'])
	
    return prompt
print(example_start_generation)