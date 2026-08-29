from langchain_neo4j import Neo4jGraph
from dataclasses import dataclass, field
from data_structures import Utterance, UtteranceType, Dialogue
from typing import List, Dict, Optional

@dataclass
class UtteranceDataset:
    """
    Dataset for managing utterance dialogues from Neo4j database.

    Args:
        n: Maximum number of utterances to extract per type
        sort: Sorting method - "date" for chronological, "random" for randomized
        text_field: Field to use for utterance text - "proposition" or "locution"
        neo4j_params: Optional dict of Neo4j connection parameters.
                     If None, uses default configuration.
        graph: Optional pre-configured Neo4jGraph instance
    """

    n: int
    sort: str = "date"
    text_field: str = "proposition"
    neo4j_params: Optional[Dict[str, str]] = None
    graph: Optional[Neo4jGraph] = None
    claims: List[Dialogue] = field(default_factory=list)
    whys: List[Dialogue] = field(default_factory=list)
    questions: List[Dialogue] = field(default_factory=list)
    concedes: List[Dialogue] = field(default_factory=list)
    sinces: List[Dialogue] = field(default_factory=list)

    def __post_init__(self):
        # Validate sort parameter
        if self.sort not in ["date", "random"]:
            raise ValueError(f"sort must be 'date' or 'random', got '{self.sort}'")

        # Validate text_field parameter
        if self.text_field not in ["proposition", "locution"]:
            raise ValueError(f"text_field must be 'proposition' or 'locution', got '{self.text_field}'")

        # Initialize graph if not provided
        if self.graph is None:
            neo4j_params = self.neo4j_params or {}
            self.graph = Neo4jGraph(**neo4j_params)
        self.load_all_utterances()
    
    def __repr__(self) -> str:
        return f"UtteranceDataset({self.counts_by_type})"

    @property
    def counts_by_type(self) -> Dict[UtteranceType, int]:
        """Get count of dialogues by utterance type."""
        return {
            UtteranceType.CLAIM: len(self.claims),
            UtteranceType.WHY: len(self.whys),
            UtteranceType.QUESTION: len(self.questions),
            UtteranceType.CONCEDE: len(self.concedes),
            UtteranceType.SINCE: len(self.sinces)
        }

    @property
    def total_count(self) -> int:
        """Get total number of dialogues across all types."""
        return len(self.claims) + len(self.whys) + len(self.questions) + len(self.concedes) + len(self.sinces)

    @property
    def available_types(self) -> List[UtteranceType]:
        """Get list of utterance types that have data."""
        types = []
        if self.claims:
            types.append(UtteranceType.CLAIM)
        if self.whys:
            types.append(UtteranceType.WHY)
        if self.questions:
            types.append(UtteranceType.QUESTION)
        if self.concedes:
            types.append(UtteranceType.CONCEDE)
        if self.sinces:
            types.append(UtteranceType.SINCE)
        return types
    

    def load_all_utterances(self) -> None:
        """Query all utterance types and store results."""
        try:
            self.claims = self._get_claim_utterances()
        except Exception as e:
            print(f"Error extracting claims: {e}")

        try:
            self.whys = self._get_why_utterances()
        except Exception as e:
            print(f"Error extracting whys: {e}")

        try:
            self.questions = self._get_question_utterances()
        except Exception as e:
            print(f"Error extracting questions: {e}")

        try:
            self.concedes = self._get_concede_utterances()
        except Exception as e:
            print(f"Error extracting concedes: {e}")

        try:
            self.sinces = self._get_since_utterances()
        except Exception as e:
            print(f"Error extracting sinces: {e}")
    
    def dialogues_of_type(self, utterance_type: UtteranceType) -> List[Dialogue]:
        """Get dialogues of a specific utterance type."""
        type_mapping = {
            UtteranceType.CLAIM: self.claims,
            UtteranceType.WHY: self.whys,
            UtteranceType.QUESTION: self.questions,
            UtteranceType.CONCEDE: self.concedes,
            UtteranceType.SINCE: self.sinces
        }
        return type_mapping.get(utterance_type, [])

    def all_final_utterances(self) -> List[Utterance]:
        """Get all final utterances from dialogues as a flattened list."""
        utterances = []
        for dialogue_list in [self.claims, self.whys, self.questions, self.concedes, self.sinces]:
            utterances.extend([dialogue.data[-1] for dialogue in dialogue_list])
        return utterances

    def all_dialogues(self) -> List[Dialogue]:
        """Get all dialogues as a flattened list."""
        all_dialogues = []
        all_dialogues.extend(self.claims)
        all_dialogues.extend(self.whys)
        all_dialogues.extend(self.questions)
        all_dialogues.extend(self.concedes)
        all_dialogues.extend(self.sinces)
        return all_dialogues
    
    
    def _get_claim_utterances(self) -> List[Dialogue]:
        order_clause = "ORDER BY claim.locution_date" if self.sort == "date" else "ORDER BY rand()"
        cypher = f"""
        MATCH (prev1)-[:TRANSITIONS_TO]->(prev2)-[:TRANSITIONS_TO]->(claim:Claim)
        WHERE NOT EXISTS {{
            MATCH (original_claim:Claim)-[:TRANSITIONS_TO]->(why:Why)-[:TRANSITIONS_TO]->(claim)
            WHERE (claim)-[:SUPPORTS]->(original_claim)
        }}
        AND NOT (prev1:Claim AND ( (claim)-[:ATTACKS]->(prev1) OR (prev1)-[:ATTACKS]->(claim) ))
        AND NOT (prev2:Claim AND ( (claim)-[:ATTACKS]->(prev2) OR (prev2)-[:ATTACKS]->(claim) ))
        WITH claim, prev1, prev2
        {order_clause}
        RETURN claim.uuid as id,
               claim.{self.text_field} as text,
               labels(claim)[0] as label,
               prev1.uuid as prev1_id,
               prev1.{self.text_field} as prev1_text,
               labels(prev1)[0] as prev1_label,
               prev2.uuid as prev2_id,
               prev2.{self.text_field} as prev2_text,
               labels(prev2)[0] as prev2_label
        LIMIT $n
        """
        return self._process_results(cypher, UtteranceType.CLAIM)
    
    def _get_why_utterances(self) -> List[Dialogue]:
        order_clause = "ORDER BY why.locution_date" if self.sort == "date" else "ORDER BY rand()"
        cypher = f"""
        MATCH (prev1)-[:TRANSITIONS_TO]->(prev2)-[:TRANSITIONS_TO]->(why:Why)
        WITH why, prev1, prev2
        {order_clause}
        RETURN why.uuid as id,
               why.{self.text_field} as text,
               labels(why)[0] as label,
               prev1.uuid as prev1_id,
               prev1.{self.text_field} as prev1_text,
               labels(prev1)[0] as prev1_label,
               prev2.uuid as prev2_id,
               prev2.{self.text_field} as prev2_text,
               labels(prev2)[0] as prev2_label
        LIMIT $n
        """
        return self._process_results(cypher, UtteranceType.WHY)
    
    def _get_question_utterances(self) -> List[Dialogue]:
        order_clause = "ORDER BY question.locution_date" if self.sort == "date" else "ORDER BY rand()"
        cypher = f"""
        MATCH (prev1)-[:TRANSITIONS_TO]->(prev2)-[:TRANSITIONS_TO]->(question:Question)
        WITH question, prev1, prev2
        {order_clause}
        RETURN question.uuid as id,
               question.{self.text_field} as text,
               labels(question)[0] as label,
               prev1.uuid as prev1_id,
               prev1.{self.text_field} as prev1_text,
               labels(prev1)[0] as prev1_label,
               prev2.uuid as prev2_id,
               prev2.{self.text_field} as prev2_text,
               labels(prev2)[0] as prev2_label
        LIMIT $n
        """
        return self._process_results(cypher, UtteranceType.QUESTION)
    
    def _get_concede_utterances(self) -> List[Dialogue]:
        order_clause = "ORDER BY concede.locution_date" if self.sort == "date" else "ORDER BY rand()"
        cypher = f"""
        MATCH (prev1)-[:TRANSITIONS_TO]->(prev2)-[:TRANSITIONS_TO]->(concede:Concede)
        WITH concede, prev1, prev2
        {order_clause}
        RETURN concede.uuid as id,
               concede.{self.text_field} as text,
               labels(concede)[0] as label,
               prev1.uuid as prev1_id,
               prev1.{self.text_field} as prev1_text,
               labels(prev1)[0] as prev1_label,
               prev2.uuid as prev2_id,
               prev2.{self.text_field} as prev2_text,
               labels(prev2)[0] as prev2_label
        LIMIT $n
        """
        return self._process_results(cypher, UtteranceType.CONCEDE)
    
    def _get_since_utterances(self) -> List[Dialogue]:
        order_clause = "ORDER BY since.locution_date" if self.sort == "date" else "ORDER BY rand()"
        cypher = f"""
        MATCH (why:Why)-[]->(claim:Claim)
        WITH why, claim
        UNWIND claim.prediction_index as index
        MATCH (since:Claim)-[:SUPPORTS]->(claim:Claim)
        {order_clause}
        RETURN since.uuid as id,
               since.{self.text_field} as text,
               labels(since)[0] as label,
               claim.uuid as prev1_id,
               claim.{self.text_field} as prev1_text,
               labels(claim)[0] as prev1_label,
               why.uuid as prev2_id,
               why.{self.text_field} as prev2_text,
               labels(why)[0] as prev2_label
        LIMIT $n
        """
        return self._process_results(cypher, UtteranceType.SINCE)
    
    def _process_results(self, cypher: str, utterance_type: UtteranceType) -> List[Dialogue]:
        results = self.graph.query(cypher, params={"n": self.n})

        dialogues = []

        for result in results:
            u0 = Utterance(
                id=str(result["prev1_id"]),
                text=result["prev1_text"],
                utterance_type=UtteranceType(result["prev1_label"].lower()),
            )
            u1 = Utterance(
                id=str(result["prev2_id"]),
                text=result["prev2_text"],
                utterance_type=UtteranceType(result["prev2_label"].lower()),
            )
            u2 = Utterance(
                id=str(result["id"]),
                text=result["text"],
                utterance_type=utterance_type,
            )
        
            dialogues.append(Dialogue(
                id=u2.id,
                data=[u0, u1, u2]
            ))

        return dialogues

    @classmethod
    def load_data(cls, dialogues: List[Dialogue]) -> 'UtteranceDataset':
        """
        Load dataset directly from list of Dialogue objects, bypassing neo4j.

        Args:
            dialogues: List of Dialogue objects to load

        Returns:
            UtteranceDataset instance with dialogues categorized by final utterance type
        """
        # Create instance without calling __init__ or __post_init__
        instance = object.__new__(cls)

        # Set minimal required attributes
        instance.n = len(dialogues)
        instance.sort = "date"
        instance.text_field = "proposition"
        instance.neo4j_params = None
        instance.graph = None
        instance.claims = []
        instance.whys = []
        instance.questions = []
        instance.concedes = []
        instance.sinces = []

        # Categorize dialogues by final utterance type
        for dialogue in dialogues:
            final_utterance_type = dialogue.data[-1].utterance_type

            if final_utterance_type == UtteranceType.CLAIM:
                instance.claims.append(dialogue)
            elif final_utterance_type == UtteranceType.WHY:
                instance.whys.append(dialogue)
            elif final_utterance_type == UtteranceType.QUESTION:
                instance.questions.append(dialogue)
            elif final_utterance_type == UtteranceType.CONCEDE:
                instance.concedes.append(dialogue)
            elif final_utterance_type == UtteranceType.SINCE:
                instance.sinces.append(dialogue)

        return instance


DUMMY_DATA = [
    # claim_claim_claim
    Dialogue(
        id="ccc",
        data=[
            Utterance(id="u1", text="The sky is blue.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u2", text="The sea is also blue.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u3", text="Neither of those things are actually blue, it's just light.", utterance_type=UtteranceType.CLAIM),
        ]
    ),
    # claim_why_since
    Dialogue(
        id="cws",
        data=[
            Utterance(id="u4", text="We should reduce carbon emissions.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u5", text="Is that really a priority?", utterance_type=UtteranceType.WHY),
            Utterance(id="u6", text="The majority of scientists agree that we should reduce carbon emissions.", utterance_type=UtteranceType.SINCE),
        ]
    ),
    # claim_why_since 2
    Dialogue(
        id="cws2",
        data=[
            Utterance(id="u19", text="We should place higher tax rates on immigrants.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u20", text="Why do you think that?", utterance_type=UtteranceType.WHY),
            Utterance(id="u21", text="Immigrants currently cost more to the economy than they provide.", utterance_type=UtteranceType.SINCE),
        ]
    ),
    # claim_question_concede
    Dialogue(
        id="cqco",
        data=[
            Utterance(id="u7", text="I think we should invest in renewable energy.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u8", text="What about the cost?", utterance_type=UtteranceType.QUESTION),
            Utterance(id="u9", text="The cost is a concern.", utterance_type=UtteranceType.CONCEDE),
        ]
    ),
    # claim_claim_why
    Dialogue(
        id="ccw",
        data=[
            Utterance(id="u10", text="Exercise is good for health.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u11", text="A balanced diet is also important.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u12", text="Why do you think that?", utterance_type=UtteranceType.WHY),
        ]
    ),
    # claim_claim_question
    Dialogue(
        id="ccq",
        data=[
            Utterance(id="u13", text="Exercise is good for health.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u14", text="A balanced diet is also important.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u15", text="What about mental health?", utterance_type=UtteranceType.QUESTION),
        ]
    ),
    # claim_claim_concede
    Dialogue(
        id="ccco",
        data=[
            Utterance(id="u16", text="Exercise is good for health.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u17", text="A balanced diet is also important.", utterance_type=UtteranceType.CLAIM),
            Utterance(id="u18", text="Yes, a balanced diet is important too.", utterance_type=UtteranceType.CONCEDE),
        ]
    ),
]