# SGD Dataset → Trace Environment Mapping

This document shows how Schema-Guided Dialogue (SGD) data maps to our RL environment for training agents.

## Example 1: Single-Domain Restaurant Dialogue

### Raw SGD Data

```json
{
  "dialogue_id": "1_00016",
  "services": ["Restaurants_1"],
  "turns": [
    {"speaker": "USER", "utterance": "I'm looking for a good place to get something to eat, can you help?"},
    {"speaker": "SYSTEM", "utterance": "Sure. What kind of food do you want? You can ask for Mexican, American..."},
    {"speaker": "USER", "utterance": "Oh, I'm in Oakland. Please find an American restaurant."},
    {"speaker": "SYSTEM", "utterance": "I'd recommend Chop Bar in Oakland.",
      "frames": [{
        "service_call": {"method": "FindRestaurants", "parameters": {"city": "Oakland", "cuisine": "American"}},
        "service_results": [{"restaurant_name": "Chop Bar", "phone_number": "510-834-2467", ...}, ...]
      }]
    },
    {"speaker": "USER", "utterance": "Are their prices extravagant? How can I contact them."},
    {"speaker": "SYSTEM", "utterance": "No, their prices are moderate. Their phone number is 510-834-2467."},
    {"speaker": "USER", "utterance": "Okay, that sounds good. Can you reserve a table for me?"},
    {"speaker": "SYSTEM", "utterance": "What time would you like it?"},
    {"speaker": "USER", "utterance": "Please reserve it for six pm."},
    {"speaker": "SYSTEM", "utterance": "Sure, please confirm that you'd like to book a table for 2 at Chop Bar..."},
    {"speaker": "USER", "utterance": "Yes that's perfect. What's their address? Do they have a live band?"},
    {"speaker": "SYSTEM", "utterance": "Your table is booked. You can find them at 247 4th Street...",
      "frames": [{
        "service_call": {"method": "ReserveRestaurant", "parameters": {...}},
        "service_results": [{"restaurant_name": "Chop Bar", "street_address": "247 4th Street", ...}]
      }]
    },
    {"speaker": "USER", "utterance": "Okay thanks. That'll be all."},
    {"speaker": "SYSTEM", "utterance": "Okay, have a good night."}
  ]
}
```

### Converted to Trace Schema

```json
{
  "conversation_id": "sgd_1_00016",
  "outcome": "resolved",
  "metadata": {
    "services": ["Restaurants_1"],
    "intents": ["FindRestaurants", "ReserveRestaurant"]
  },
  "messages": [
    {"role": "user", "content": "I'm looking for a good place to get something to eat, can you help?"},
    {"role": "assistant", "content": "Sure. What kind of food do you want? You can ask for Mexican, American..."},
    {"role": "user", "content": "Oh, I'm in Oakland. Please find an American restaurant."},
    {"role": "assistant", "content": "I'd recommend Chop Bar in Oakland.",
      "tool_calls": [{"name": "FindRestaurants", "args": {"city": "Oakland", "cuisine": "American"}}]
    },
    {"role": "tool", "name": "FindRestaurants", "result": [
      {"restaurant_name": "Chop Bar", "phone_number": "510-834-2467", "price_range": "moderate", ...}
    ]},
    {"role": "user", "content": "Are their prices extravagant? How can I contact them."},
    {"role": "assistant", "content": "No, their prices are moderate. Their phone number is 510-834-2467."},
    ...
  ]
}
```

---

## RL Environment Execution

Here's how the environment uses this trace for training:

### Step-by-Step Episode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ EPISODE START: Trace "sgd_1_00016" (Restaurant Booking)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ env.reset() →                                                               │
│   observation = {                                                           │
│     "user_message": "I'm looking for a good place to eat...",               │
│     "available_tools": ["FindRestaurants", "ReserveRestaurant"],            │
│     "history": []                                                           │
│   }                                                                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ STEP 1: Agent responds (no tool call needed)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ agent_action = "What city and cuisine type would you prefer?"               │
│                                                                             │
│ env.step(agent_action) →                                                    │
│   ground_truth = "Sure. What kind of food do you want?..."                  │
│   reward = semantic_similarity(agent_action, ground_truth)  # e.g., 0.85    │
│                                                                             │
│   observation = {                                                           │
│     "user_message": "Oh, I'm in Oakland. Please find an American restaurant."│
│     "history": [prev_turns...]                                              │
│   }                                                                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ STEP 2: Agent should call a tool                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ agent_action = {                                                            │
│   "response": "Let me search for American restaurants in Oakland.",         │
│   "tool_call": {"name": "FindRestaurants", "args": {"city": "Oakland", ...}}│
│ }                                                                           │
│                                                                             │
│ → tool_mocker returns mocked results based on ground truth                  │
│ → reward components:                                                        │
│     - tool_selection_reward: +1.0 (correct tool chosen)                     │
│     - argument_accuracy: 0.9 (correct args)                                 │
│     - response_quality: 0.8                                                 │
│                                                                             │
│   observation = {                                                           │
│     "user_message": "Are their prices extravagant?...",                     │
│     "tool_result": [{"restaurant_name": "Chop Bar", ...}],                  │
│     "history": [...]                                                        │
│   }                                                                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ ... continues until conversation ends ...                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ EPISODE END                                                                 │
│   total_reward = sum(step_rewards)                                          │
│   episode_success = (final_action == "GOODBYE") && (reservation_made)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example 2: Multi-Domain Dialogue (RentalCars + Movies)

This shows a more complex trace spanning multiple services:

### Conversation Flow

```
USER: I need to rent a compact car starting tomorrow in SFO
      ↓
SYSTEM: [Calls GetCarsAvailable] → Returns Chevrolet Bolt for $69
      ↓
USER: Book it for pickup at noon, returning March 5th
      ↓
SYSTEM: [Calls ReserveCar] → Confirms reservation
      ↓
USER: Thanks. Can you also find me a movie at Century San Francisco Centre?  ← DOMAIN SWITCH
      ↓
SYSTEM: [Calls FindMovies] → Returns Captain Marvel, Hellboy, Missing Link
      ↓
USER: Actually, what nature preserves are good for kids in SF?  ← ANOTHER DOMAIN SWITCH
      ↓
SYSTEM: [Calls FindAttractions] → Returns parks/preserves
```

### Key RL Training Signals

| Turn | Ground Truth Action | Training Signal |
|------|---------------------|-----------------|
| 1 | REQUEST slots (city, dates) | Agent learns to gather info |
| 3 | Call `GetCarsAvailable` | +reward for correct tool |
| 5 | Call `ReserveCar` | +reward for tool + correct args |
| 7 | Call `FindMovies` (domain switch!) | Agent learns multi-domain handling |
| 9 | Call `FindAttractions` | Validates generalization |

---

## Reward Signal Components

The SGD dataset provides rich signals for computing rewards:

### 1. **Response Quality** (from utterances)
```python
reward = semantic_similarity(
    agent_response,
    ground_truth_utterance
)
```

### 2. **Tool Selection** (from service_call.method)
```python
if agent_tool == ground_truth_tool:
    reward += 1.0
elif agent_tool in valid_alternatives:
    reward += 0.5
else:
    reward -= 0.5
```

### 3. **Argument Accuracy** (from service_call.parameters)
```python
correct_args = set(agent_args.items()) & set(gt_args.items())
reward += len(correct_args) / len(gt_args)
```

### 4. **Slot Extraction** (from state.slot_values)
```python
# User says: "I'm in Oakland"
# Ground truth slot: {"city": ["Oakland"]}
# If agent extracts city correctly → +reward
extracted_slots = agent.extract_slots(user_utterance)
reward += f1_score(extracted_slots, gt_slot_values)
```

### 5. **Intent Tracking** (from state.active_intent)
```python
# Reward for maintaining correct intent understanding
if agent_intent == ground_truth_intent:
    reward += 0.3
```

### 6. **Outcome Success** (from action types)
```python
# NOTIFY_SUCCESS in ground truth → episode should end successfully
if gt_action == "NOTIFY_SUCCESS" and agent_completed_task:
    reward += 2.0
elif gt_action == "NOTIFY_FAILURE" and agent_handled_gracefully:
    reward += 1.0
```

---

## Tool Mocking Strategy

The `tool_mocker` uses `service_results` from the trace:

```python
class SGDToolMocker:
    def __init__(self, trace):
        # Pre-index all service results from the trace
        self.service_results = {}
        for turn in trace["turns"]:
            for frame in turn.get("frames", []):
                if "service_call" in frame:
                    method = frame["service_call"]["method"]
                    params = frame["service_call"]["parameters"]
                    results = frame.get("service_results", [])
                    self.service_results[(method, frozenset(params.items()))] = results
    
    def call(self, tool_name: str, args: dict) -> list:
        # Fuzzy match against known calls
        key = (tool_name, frozenset(args.items()))
        if key in self.service_results:
            return self.service_results[key]
        
        # Fallback: find closest matching call
        return self._fuzzy_match(tool_name, args)
```

---

## Environment State Space

Based on SGD structure, the observation space includes:

```python
@dataclass
class Observation:
    user_message: str                    # Current user utterance
    history: List[Message]               # Conversation so far
    available_tools: List[ToolSchema]    # From schema.json
    tool_result: Optional[Any]           # If previous action was tool call
    
    # Derived from SGD state tracking (for evaluation)
    active_intent: str                   # Ground truth intent
    slot_values: Dict[str, List[str]]    # Extracted entities
    requested_slots: List[str]           # What user is asking for
```

---

## Action Space

```python
@dataclass
class AgentAction:
    response: str                                    # Natural language response
    tool_call: Optional[ToolCall] = None            # Optional tool invocation
    
@dataclass  
class ToolCall:
    name: str                           # Tool/API name (e.g., "FindRestaurants")
    args: Dict[str, Any]                # Parameters
```

---

## Training Strategy Recommendations

### Phase 1: Single-Domain Training
- Use ~7,500 single-domain dialogues
- Focus on: slot extraction, tool selection, response generation
- Simpler reward signal

### Phase 2: Multi-Domain Training  
- Add 15,000+ multi-domain dialogues
- Focus on: domain switching, context retention, multiple tool chains
- Curriculum learning: sort by # of domains

### Phase 3: Zero-Shot Evaluation
- Test set has **unseen domains** (e.g., new API schemas)
- Evaluates generalization to new tools

### Difficulty Curriculum

```python
def sort_by_complexity(dialogues):
    return sorted(dialogues, key=lambda d: (
        len(d["services"]),           # Multi-domain harder
        count_service_calls(d),       # More tool calls harder
        len(d["turns"]),              # Longer conversations harder
    ))
```

