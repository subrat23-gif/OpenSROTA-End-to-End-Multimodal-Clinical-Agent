"""Clinical state schema. One uniform Fact type keeps extraction, merging and evaluation simple.

Key design points vs. the original plan:
  * `assertion` covers more than negation (absent / uncertain / hypothetical).
  * `experiencer` separates patient facts from family history ("my father has diabetes").
  * Evidence is (turn_id, verbatim quote). The LLM NEVER emits timestamps; code derives them
    from turn ids, so they cannot be hallucinated.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

FIELDS = ["complaints", "symptoms", "history", "medications", "allergies", "vitals",
          "examination", "assessment", "investigations", "advice", "follow_up", "referrals"]


class Assertion(str, Enum):
    present = "present"
    absent = "absent"            # explicitly denied
    uncertain = "uncertain"      # "maybe", "not sure"
    hypothetical = "hypothetical"  # "if the pain gets worse..."


class Experiencer(str, Enum):
    patient = "patient"
    family = "family"
    other = "other"


class Source(str, Enum):
    patient_reported = "patient_reported"
    doctor_observed = "doctor_observed"
    doctor_assessed = "doctor_assessed"
    explicit_plan = "explicit_plan"
    model_uncertain = "model_uncertain"


class Attribute(BaseModel):
    key: str    # e.g. duration, severity, dose, frequency, trigger
    value: str


class Evidence(BaseModel):
    turn_id: int
    quote: str  # verbatim, <= ~20 words


class Turn(BaseModel):
    id: int
    speaker: str  # "Doctor" | "Patient"
    text: str
    start: float = 0.0
    end: float = 0.0


class Fact(BaseModel):
    """What the LLM is asked to produce."""
    name: str
    attributes: List[Attribute] = Field(default_factory=list)
    assertion: Assertion = Assertion.present
    experiencer: Experiencer = Experiencer.patient
    source: Source
    status: Optional[str] = None  # ordered/recommended/completed/pending/discussed/current/discontinued
    evidence: List[Evidence] = Field(default_factory=list)


class StoredFact(Fact):
    """Fact plus fields filled in by code (never by the LLM)."""
    supported: Optional[bool] = None
    weak_link: Optional[bool] = None  # every cited quote is <= 1 word (soft flag)
    name_ungrounded: Optional[bool] = None   # no content token of `name` occurs in any cited quote
    value_ungrounded: Optional[bool] = None  # an attribute value contains a number absent from the quotes
    t_start: Optional[float] = None
    t_end: Optional[float] = None


class Extraction(BaseModel):
    complaints: List[Fact] = Field(default_factory=list)
    symptoms: List[Fact] = Field(default_factory=list)
    history: List[Fact] = Field(default_factory=list)
    medications: List[Fact] = Field(default_factory=list)
    allergies: List[Fact] = Field(default_factory=list)
    vitals: List[Fact] = Field(default_factory=list)
    examination: List[Fact] = Field(default_factory=list)
    assessment: List[Fact] = Field(default_factory=list)
    investigations: List[Fact] = Field(default_factory=list)
    advice: List[Fact] = Field(default_factory=list)
    follow_up: List[Fact] = Field(default_factory=list)
    referrals: List[Fact] = Field(default_factory=list)


class ClinicalState(BaseModel):
    complaints: List[StoredFact] = Field(default_factory=list)
    symptoms: List[StoredFact] = Field(default_factory=list)
    history: List[StoredFact] = Field(default_factory=list)
    medications: List[StoredFact] = Field(default_factory=list)
    allergies: List[StoredFact] = Field(default_factory=list)
    vitals: List[StoredFact] = Field(default_factory=list)
    examination: List[StoredFact] = Field(default_factory=list)
    assessment: List[StoredFact] = Field(default_factory=list)
    investigations: List[StoredFact] = Field(default_factory=list)
    advice: List[StoredFact] = Field(default_factory=list)
    follow_up: List[StoredFact] = Field(default_factory=list)
    referrals: List[StoredFact] = Field(default_factory=list)