# from typing import TypedDict

# from persuasio.states.state import HumanState, GenerationAgentsState, BaseModelState

def is_typed_dict_type(cls):
    return isinstance(cls, type) and issubclass(cls, dict) and hasattr(cls, "__annotations__") and (
        hasattr(cls, "__total__") or "__required_keys__" in cls.__dict__
    )    
    
def is_instance_of_typed_dict(obj, typed_dict_cls):
    if not is_typed_dict_type(typed_dict_cls):
        raise TypeError(f"{typed_dict_cls} is not a TypedDict type.")
    return isinstance(obj, dict) and all(
        key in obj for key in typed_dict_cls.__annotations__
    )