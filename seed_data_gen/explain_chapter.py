base_instruction = "Imagine a Social Story book designed as an effective and meaningful approach to help children with Autism Spectrum Disorder (ASD) understand and navigate various social situations through stories. \
The ultimate and deeper goal is to empower children and older people by enhancing their understanding of social situations and social encounters in their lives, and thereby supporting their ability to be active participants in life’s routines and activities. \n\
The book contains chapters on different themes that are crucial for social interaction, emotional understanding and so on. \n"


# "I want you act as a Prompt Creator.\r\n\
# Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
# This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
# The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
# The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
# '#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"



def createExplainChapterPrompt(chapter_title_dict):
	prompt = base_instruction
	prompt += "# Given a chapter and story titles in the chapter as below: # \r\n {} \r\n".format(chapter_title_dict)
	prompt += "Please use one concise sentence(begaining with a verb) to explain the chapter's focus, \
	which highlight how the chapter supports ASD children in developing corresponding critical skills and understandings. \
	You must just give the explanation without any other outputs. \
	'The explanation', 'The explanation:' are not allowed to appear in #Rewritten Prompt#\n"
	prompt += "#The Explanation: # \r\n"
	return prompt


prompt = createExplainChapterPrompt("=======================")
print(prompt)