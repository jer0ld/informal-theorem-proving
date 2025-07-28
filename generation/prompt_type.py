from enum import Enum

class PromptType(Enum):
  ZERO_SHOT = 0
  CHAIN_OF_THOUGHT = 1
  FEW_SHOT = 2