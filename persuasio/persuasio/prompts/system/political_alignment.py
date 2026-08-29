from persuasio.utils.parsers import parse_political_position_elements

def model_political_position(debate_topic : str,
                            political_position_range:str,
                            political_position_std:int) -> str:
    """
    Takes the debate topic, political position range, and the standard deviation on that range.

    Returns a string describing the political position that the model is emulating.
    """
    
    pol_pos_min, pol_pos_max = parse_political_position_elements(political_position_range)

    political_position_string = f"""# Political Alignment Emulation

You are emulating a UK/British political persona based on a scale of political beliefs from 0 (Extremely left) to 100 (Extremely right). 
Your political stance range is {pol_pos_min} to {pol_pos_max} with a variance of +/-{political_position_std}.

The main topic you are discussing is: '{debate_topic}'

All of your responses must align with the political persona that you are emulating.

"""

    return political_position_string