import os 

# For OpenAI's models
from openai import OpenAI
from openai import APIError, APIConnectionError, Timeout

# For the retry logic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setting up LangSmith tracing to understand what's happening in our models under the hood.
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable

from persuasio.datatypes.enums import ModelName


# Define a retry decorator for OpenAI calls
openai_retry = retry(
    retry=retry_if_exception_type((Timeout, APIConnectionError, APIError)),
    stop=stop_after_attempt(5),              # Retry up to 3 times
    wait=wait_exponential(multiplier=2, max=32),  # Exponential backoff: 2s, 4s, 8s..., max 20s
    reraise=True                             # Raise exception if all retries fail
)

TIMEOUT = 120

class GenerateLLMResponses:

    def __init__(self,model_choice,prompt,temperature,top_p,seed,datatype_schema):

        self.model_choice = model_choice
        self.prompt = prompt
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.datatype_schema = datatype_schema    
        

        self.client, self.model = self.llm_client()

        # Auto-trace LLM calls in-context for debugging with LangSmith
        # if (self.model != "llama3.1:8b") and (self.model != "llama3.2:3b"):
        #     self.client = wrap_openai(self.client)

        


    def llm_client(self):

        """Takes the model chosen by the user and sets up the AzureOpenAI API key, endpoint, and api version for the following models:
            
            - gpt-4o-mini (CustomContentFilter == off) <- for testing
            - gpt-5 (CustomContentFilter == off)
            - deepseek-r1 (CustomContentFilter == off) (still need to implement)
            - gemini-25-pro (still need to implement)
            - mistral-medium (CustomContentFilter == off)
            - grok-3 (CustomContentFilter == off)
            
        Returns: the client API which will be called to make completions/generations and the model name, used when running the following function:
            - client.chat.completions.create() 
        """
        
        
        if self.model_choice == ModelName.GPT_4O_MINI:

            self.client = OpenAI(
                base_url=os.environ.get("AZURE_OPENAI_GPT4oMini_ENDPOINT"),
                api_key = os.environ.get("AZURE_OPENAI_GPT4oMini_API_KEY"),
                timeout=TIMEOUT
            )

            self.model = ModelName.GPT_4O_MINI.value
        
        elif self.model_choice == ModelName.GPT_5:

            self.client = OpenAI(
                base_url=os.environ.get("AZURE_OPENAI_GPT5_CHAT_API_KEY"),
                api_key = os.environ.get("AZURE_OPENAI_GPT5_CHAT_API_KEY"),
                timeout=TIMEOUT
            )

            self.model = ModelName.GPT_5.value + "-chat"

        elif self.model_choice == ModelName.GPT_4O:

            self.client = OpenAI(
                base_url=os.environ.get("AZURE_OPENAI_GPT4O_ENDPOINT"),
                api_key = os.environ.get("AZURE_OPENAI_GPT4O_API_KEY"),
                timeout=TIMEOUT
            )

            self.model = ModelName.GPT_4O.value

        elif self.model_choice == ModelName.GROK_3:

            self.client = OpenAI(
                base_url=os.environ.get("AZURE_GROK_3_ENDPOINT"),
                api_key = os.environ.get("AZURE_GROK_3_API_KEY"),
                timeout=TIMEOUT
            )

            self.model = ModelName.GROK_3.value

        elif self.model_choice == ModelName.GROK_4:

            self.client = OpenAI(
                base_url=os.environ.get("AZURE_GROK_4_ENDPOINT"),
                api_key = os.environ.get("AZURE_GROK_4_API_KEY"),
                timeout=TIMEOUT
            )

            self.model = ModelName.GROK_4.value + "-fast-non-reasoning"

        elif self.model_choice == ModelName.MISTRAL_MEDIUM:

            self.client = OpenAI(
                base_url=os.environ.get("AZURE_MISTRAL_MEDIUM_ENDPOINT"),
                api_key = os.environ.get("AZURE_MISTRAL_MEDIUM_API_KEY"),
                timeout=TIMEOUT
            )

            self.model = ModelName.MISTRAL_MEDIUM.value + "-2505"

        return self.client, self.model
    
    
    #@traceable # Auto-trace this function
    @openai_retry
    def return_completion(self):
            
        schema = self.datatype_schema

        if self.model_choice == ModelName.GPT_5:
            self.completion = self.client.beta.chat.completions.parse(
                model = self.model,
                messages=self.prompt,
                seed=self.seed,
                response_format=schema
            )
        
        elif (self.model_choice == ModelName.GPT_4O_MINI) or (self.model_choice == ModelName.GPT_4O):
            # Get JSON Schema from Pydantic model
            schema = self.datatype_schema.model_json_schema()
            
            # Ensure required matches exactly the keys in properties
            schema["required"] = list(schema["properties"].keys())

            self.completion = self.client.beta.chat.completions.parse(
                model = self.model,
                messages=self.prompt,
                temperature = self.temperature,
                top_p=self.top_p,
                seed=self.seed,
                response_format={
                    "type": "json_schema",
                    "json_schema" : {
                        "name" : self.datatype_schema.__name__,
                        "schema" : schema,
                    }
                    
                }
            )

            # raw_output = self.completion.choices[0].message.content

            # try:
            #     # try to return JSON
            #     validated = self.datatype_schema.model_validate_json(raw_output)
                
            # except Exception as e:
            #     # if it fails, then fall back to SafeBaseModel handling
            #     validated = self.datatype_schema.from_llm(raw_output)

            # return validated.dict()




        else:
            self.completion = self.client.beta.chat.completions.parse(
                model = self.model,
                messages=self.prompt,
                temperature = self.temperature,
                top_p=self.top_p,
                seed=self.seed,
                response_format=schema
            )

        raw_output = None

        # Try to extract message content or parsed object safely
        choice = self.completion.choices[0].message
        if hasattr(choice, "parsed") and choice.parsed is not None:
            try:
                return choice.parsed.model_dump()
            except Exception:
                # fallback to text content if parsing failed
                raw_output = getattr(choice, "content", None)
        else:
            raw_output = getattr(choice, "content", None)

        if raw_output is None:
            raise ValueError("Model returned no output to parse.")

        raw_output = raw_output.strip()
        
        if raw_output.startswith("{") and not raw_output.endswith("}"):
            raw_output = raw_output + "}"

        try:
            validated = self.datatype_schema.model_validate_json(raw_output)
        except Exception:
            validated = self.datatype_schema.from_llm(raw_output)

        return validated.dict()

        # return self.completion.choices[0].message.parsed.model_dump()