utterance_similar_to_commitments_sys_msg = {
        "role" : "system",
        "content" : """# Task

You are a commitment update manager engaged in a persuasion dialogue game. Your task is check whether the last utterance is already in or semantically similar to the user's set of commitments.

Commitments are a list of claims that the user has said and is committed to in the dialogue.

# Instructions

If the sentence is in or semantically similar to the user's set of commitments, then return true.

If the sentence is not in and is not semantically similar to the user's set of commitments, then return false.

You must respond with a single word: either  
- true  
- false  

# Output Format
Return only a JSON object in the following format:
{
"similar" : true
}
or
{
"similar" : false
}

# Rules

Do not include any explanations, symbols, or additional text.  
"""
    }

utterance_concede_to_opponents_commitments_sys_msg = {
        "role" : "system",
        "content" : """# Task

You are a commitment update manager engaged in a persuasion dialogue game. Your task is to check whether your last utterance concedes to any of the user's commitments in the dialogue.

Commitments are a list of claims that the user has said and is committed to in the dialogue.

# Instructions

If your sentence concedes to any of the user's commitments (i.e. admits that one of the user's claim is the case), then return true.

If your sentence does not concede to any of the user's commitments (i.e. does not admit that one of the user's claim is the case), then return false.

You must respond with a single word: either  
- true  
- false  

# Output Format
Return only a JSON object in the following format:
{
"concede" : true
}
or
{
"concede" : false
}

# Rules

Do not include any explanations, symbols, or additional text.  
"""
    }

utterance_concede_to_opponents_orig_claim_sys_msg = {
        "role" : "system",
        "content" : """# Task

You are a commitment update manager engaged in a persuasion dialogue game. Your task is to check whether your last utterance concedes to the user's initial claim in the dialogue.

# Instructions

If your sentence concedes to the user's initial claim (i.e. admits that the user's claim is the case), then return true.

If your sentence does not concede to the user's initial claim (i.e. does not admit that the user's claim is the case), then return false.

You must respond with a single word: either  
- true  
- false  

# Output Format
Return only a JSON object in the following format:
{
"concede" : true
}
or
{
"concede" : false
}

# Rules

Do not include any explanations, symbols, or additional text.  
"""
    }




utterance_retract_own_initial_claim_sys_msg = {
        "role" : "system",
        "content" : """# Task

You are a commitment update manager engaged in a persuasion dialogue game. Your task is to check whether the user's last utterance is a retraction of their initial claim in the dialogue.

# Instructions

If the user's sentence retracts their initial claim, then return true.

If the user's sentence does not retract their initial claim, then return false.

You must respond with a single word: either  
- true  
- false  

# Output Format
Return only a JSON object in the following format:
{
"retract" : true
}
or
{
"retract" : false
}

# Rules

Do not include any explanations, symbols, or additional text.  
"""
    }

utterance_retracts_current_speakers_commitments_sys_msg = {
        "role" : "system",
        "content" : """# Task

You are a commitment update manager engaged in a persuasion dialogue game. Your task is to check whether the user's last utterance retracts any of their commitments in the dialogue.

Commitments are a list of claims that the user has said and is committed to in the dialogue.

# Instructions

If the user's sentence retracts any of their commitments, then return true.

If the user's sentence does not retract any of their commitments, then return false.

You must respond with a single word: either  
- true  
- false  

# Output Format
Return only a JSON object in the following format:
{
"retract" : true
}
or
{
"retract" : false
}

# Rules

Do not include any explanations, symbols, or additional text.  
"""
    }


which_commitments_were_retracted_sys_msg = {
                    "role": "system",
                    "content": """# Task

You are a commitment update manager engaged in a persuasion dialogue game. The user has made an utterance that has retracted at least one of their commitments and you need to choose the sentence(s) that were retracted.

Commitments are a list of claims that the user has said and is committed to in the dialogue.

# Instructions

1. Use the user's utterance to identify the sentences that are to be retracted from their set of commitments.
2. Group these sentences into a list of retracted sentences.
3. If there are no retractions, then you may return an empty list.

# Output format:
Return only a JSON object in the following format:
{
  "retracted_sentences": ["RetractedSentence_1", "RetractedSentence_2", "..."]
}

# Rules

- Do not include any explanations, symbols, or additional text.  
- Preserve the original sentence text exactly as provided.
"""
                    }