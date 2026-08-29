import os
import asyncio
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI
from typing import Optional, List
from pydantic import BaseModel
from collections import Counter

from data_structures import UtteranceType, Utterance, Dialogue, Prediction, ModelParameters

class Classification(BaseModel):
    Classification: UtteranceType

class UtteranceClassifier:
    def __init__(
        self, 
        model: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-08-01-preview"):
        
        self.model = model
        
        if endpoint and "azure" in endpoint.lower():
            self.client = AzureOpenAI(
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=endpoint,
                api_version=api_version
            )
            self.async_client = AsyncAzureOpenAI(
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=endpoint,
                api_version=api_version
            )
        else:
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=endpoint
            )
            self.async_client = AsyncOpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=endpoint
            )
    

    def _build_api_params(self, dialogue: Dialogue, model_params: ModelParameters) -> dict:
        """Build API parameters from dialogue and model parameters"""
        prompt = dialogue.prompt()

        api_params = {
            "model": self.model,
            "messages": prompt,
            "response_format": Classification,
            "temperature": model_params.temperature
        }

        # Add optional parameters if specified
        if model_params.seed is not None:
            api_params["seed"] = model_params.seed

        return api_params

    def _parse_response(self, response) -> UtteranceType:
        """Parse API response and extract classification"""
        parsed_result = response.choices[0].message.parsed
        return parsed_result.Classification

    def classify(self, dialogue: Dialogue, model_params: ModelParameters = None) -> Prediction:
        """Classify a single utterance and return prediction"""
        if model_params is None:
            model_params = ModelParameters()

        try:
            api_params = self._build_api_params(dialogue, model_params)
            response = self.client.beta.chat.completions.parse(**api_params)
            predicted_type = self._parse_response(response)

            return Prediction(
                utterance=dialogue.data[-1],
                dialogue=dialogue,
                label=predicted_type,
                model_name=self.model,
                model_params=model_params
            )

        except Exception as e:
            print(f"Error classifying utterance: {e}")
            return None
    
    async def classify_async(self, dialogue: Dialogue, model_params: ModelParameters = None) -> Prediction:
        """Async version of classify for parallel processing"""
        if model_params is None:
            model_params = ModelParameters()

        try:
            api_params = self._build_api_params(dialogue, model_params)
            response = await self.async_client.beta.chat.completions.parse(**api_params)
            predicted_type = self._parse_response(response)

            return Prediction(
                utterance=dialogue.data[-1],
                dialogue=dialogue,
                label=predicted_type,
                model_name=self.model,
                model_params=model_params
            )

        except Exception as e:
            print(f"Error classifying utterance: {e}")
            return None

    def _compute_majority_vote(self, predictions: List[UtteranceType]) -> UtteranceType:
        """Compute majority vote from a list of predictions"""
        if not predictions:
            raise ValueError("Cannot compute majority vote from empty predictions list")

        counter = Counter(predictions)
        # Get the most common prediction (first in case of tie)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def classify_with_repetitions(self,
        dialogue: Dialogue,
        repetitions: int = 5,
        params: ModelParameters = None,
        seed_start: int = 100) -> Prediction:
        """Classify with multiple repetitions and return majority vote"""

        if params is None:
            params = ModelParameters()

        predictions = []
        seeds = []
        original_seed = params.seed

        for i in range(repetitions):
            seed = seed_start + i
            seeds.append(seed)

            # Modify seed for this repetition (avoid creating new object)
            params.seed = seed
            prediction = self.classify(dialogue, params)
            if prediction is not None:
                predictions.append(prediction.label)

        # Restore original seed
        params.seed = original_seed

        if not predictions:
            return None

        # Compute majority vote
        majority_prediction = self._compute_majority_vote(predictions)

        return Prediction(
            utterance=dialogue.data[-1],
            dialogue=dialogue,
            label=majority_prediction,
            model_name=self.model,
            model_params=params,
            r_labels=predictions,
            individual_seeds=seeds
        )


    async def classify_dialogues_batch(self,
                                      dialogues: list[Dialogue],
                                      max_concurrent: int = 10,
                                      model_params: ModelParameters = None,
                                      show_progress: bool = False) -> list[Prediction]:
        """Classify multiple dialogues in parallel with concurrency limit"""
        if model_params is None:
            model_params = ModelParameters()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def classify_with_semaphore(dialogue):
            async with semaphore:
                return await self.classify_async(dialogue, model_params)

        tasks = [classify_with_semaphore(dialogue) for dialogue in dialogues]

        if show_progress:
            try:
                from tqdm.asyncio import tqdm
                results = await tqdm.gather(*tasks, desc="Classifying dialogues")
            except ImportError:
                print("tqdm not available, running without progress bar")
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                # If any task fails with tqdm, fall back to regular gather
                results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Exception classifying utterance {i}: {result}")
            elif result is not None:
                valid_results.append(result)

        return valid_results

    async def classify_dialogues_with_repetitions_batch(self,
                                                       dialogues: list[Dialogue],
                                                       repetitions: int = 5,
                                                       max_concurrent: int = 10,
                                                       params: ModelParameters = None,
                                                       seed_start: int = 100,
                                                       show_progress: bool = False) -> list[Prediction]:
        """Classify multiple dialogues with repetitions using async batch with sync repetitions"""
        if params is None:
            params = ModelParameters()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def classify_dialogue_with_repetitions(dialogue, dialogue_seed_start):
            async with semaphore:
                # Use sync repetition method within async context
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.classify_with_repetitions,
                    dialogue,
                    repetitions,
                    params,
                    dialogue_seed_start
                )

        # Create tasks for each dialogue with different seed ranges
        tasks = []
        for i, dialogue in enumerate(dialogues):
            # Each dialogue gets its own seed range to avoid overlap
            dialogue_seed_start = seed_start + (i * repetitions)
            tasks.append(classify_dialogue_with_repetitions(dialogue, dialogue_seed_start))

        if show_progress:
            try:
                from tqdm.asyncio import tqdm
                results = await tqdm.gather(*tasks, desc=f"Classifying dialogues ({repetitions} reps each)")
            except ImportError:
                print("tqdm not available, running without progress bar")
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                # If any task fails with tqdm, fall back to regular gather
                results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Exception classifying dialogue {i} with repetitions: {result}")
            elif result is not None:
                valid_results.append(result)

        return valid_results
