from typing import List, Dict

from persuasio.prompts.system.disambiguation import (
  max_sentence_reached_sys_msg,
  group_similar_sents_sys_msg, 
  choose_one_sent_sys_msg)

def are_sentences_in_list_similar(sentences : List[str]) -> List[List[str]]:
  
  human_msg = {"role" : "user",
                     "content" : """Here is the list of sentences:

"""+"\n".join(sentences)+"""

Identify and group the sentences with the same or highly similar semantic meaning, following the rules in the system instructions."""}

  prompt = [group_similar_sents_sys_msg, human_msg]
        
  return prompt
    
def choose_one_sentence(sentences : List[str]) -> List[Dict[str,str]]:

  human_msg = {"role" : "user",
                "content" : """Here is the list of sentences:

"""+"\n".join(sentences)+"""

Choose the most persuasive sentence, following the rules in the system instructions."""}
  
  prompt = [choose_one_sent_sys_msg, human_msg]

  return prompt


def choose_most_persuasive_sents_to_return(sentences : str, max_sentences : int) -> List[Dict[str, str]]:

  sentences = "\n".join(sentences)

  sys_msg = max_sentence_reached_sys_msg(max_sentences=max_sentences)

  human_msg = {
    "role" : "user",
    "content" : f"""Here is the list of sentences to choose from:

{sentences}

Choose the {max_sentences} most persuasive sentences.
"""
  }

  prompt = [sys_msg, human_msg]

  return prompt