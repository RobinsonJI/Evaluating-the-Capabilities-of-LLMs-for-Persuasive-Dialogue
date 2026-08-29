from typing import Dict

group_similar_sents_sys_msg = {
                    "role": "system",
                    "content": """# Task
                    
You are a helpful assistant that is conducting sentence disambiguation.

# Instructions

1. Identify all sentences that convey the same or very similar meaning at the sentence level.
2. Group these sentences into a 2D array (a list of lists), where each inner list contains sentences sharing a single distinct semantic meaning.
3. Exclude any sentences that do not have at least one other sentence with a similar meaning (no groups of size one).


# Output format:
Return only a JSON object in the following format:
{
  "ListOfSimilarSentences": [
    ["similarSentenceA_1", "similarSentenceA_2", "..."],
    ["similarSentenceB_1", "similarSentenceB_2", "..."],
    ...
  ]
}

# Rules:
- Do not include any explanation, commentary, or additional text outside the JSON.
- Preserve the original sentence text exactly as provided.
"""
                    }

choose_one_sent_sys_msg = {
            "role" : "system",
            "content" : """# Task
            
You are a helpful assistant given a list of sentences. Your task is to choose the most representative sentence from the list of sentences that you believe is the most persuasive.

# Output format:
Return only a JSON object in the following format:
{
  "Choice": str
}

# Rules:
- Do not include any explanation, commentary, or additional text outside the JSON.
- Preserve the original sentence text exactly as provided."""
        }


def max_sentence_reached_sys_msg(max_sentences : int) -> Dict[str, str]:
  disambig_persuasiveness_choice_sys_msg = {
      "role" : "system",
      "content" : f"""# Task

You have been given a list of sentences. Your job it is to choose the {max_sentences} most persuasive sentences to return to the user.

A persuasive sentence is one that is more likely to convince someone, inspire action, or change their mind. 
Use the following criteria to rank the sentences: tone, clarity, emotional impact, word choice, and call to action.
Persuasive sentences are: clear; confident; and encourage action.

# Instructions

- Read the sentences from the user.

- Choose the {max_sentences} most persuasive sentences."""+"""

# Output format:
Return only a JSON object in the following format:
{
  "Choices": ["PersuasiveSentence_1", "PersuasiveSentence_2", "...", """+f""" "PersuasiveSentence_{max_sentences}" """+"""]
}

# Rules:
- Do not include any explanation, commentary, or additional text outside the JSON.
- Preserve the original sentence text exactly as provided."""
  }
  return disambig_persuasiveness_choice_sys_msg