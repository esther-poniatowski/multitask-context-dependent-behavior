"""
`core.parse.extract_events` [module]

Notes
-----
The input arrays contain one element per event. They specify the following information:

- ``block``:   Number of the block within the session (denoted as "trial" in the raw data)
- ``t_start``: Start time of the event (in seconds) relative to the start of the session
- ``t_end``:   Stop time of the event (in seconds) relative to the start of the session
- ``description``: Description of the event in the raw data

The ``description`` field is a string which can comprise several details separated by commas.

1. Type of the event (always present), among a set of predefined values : "TRIALSTART",
  "PreStimSilence", "Stim", "PostStimSilence", "BEHAVIOR", "SHOCKON", "TRIALSTOP".
2. Stimulus identity (if the first is PreStimSilence, Stim or PostStimSilence). Example:
   'TORC_448_06_v501', '2000', '8000'...
3. Stimulus category (if applicable) : 'Reference' or 'Target'.
4. Stimulus sound intensity (if applicable), for instance '0dB', '10dB'...

Examples of strings that can be encountered :

- ``'PreStimSilence , TORC_448_06_v501 , Reference'`` (3 elements)
- ``'TRIALSTART'`` (single element)

The output arrays should contain one element per slot. See the `SessionParser` class for more
details.


In the raw data, the events are organized in blocks, each block containing a sequence of slots. The
boundaries to group events (in blocks or slots) can be identified from the events descriptions:

- The boundaries of one block are delimited by the keywords ``TRIALSTART`` and ``TRIALSTOP``.
- The boundaries of one slot are delimited by the keyword ``PreStimSilence``.

Implementation
--------------
Data Structures:
Several classes are defined to separate concerns for the different hierarchical level in the data:

- EventManager: Represents individual events from the raw data.
- SlotManager: Represents slots within blocks.
- BlockManager: Represents blocks containing slots.
- SessionParser: Main class for processing the entire session.

Processing Flow:
1. Initialize SessionParser with new raw event data.
2. Convert raw events to EventManager objects.
3. Iterate through events:
    - Start a new block on TRIALSTART events.
    - End the current block on TRIALSTOP events.
    - Process other events within the current block:
    * Start a new slot on PreStimSilence events.
    * Update slot information for Stim, PostStimSilence, and other events.
4. Aggregate results from all blocks and slots.

Output: The parser returns numpy arrays for slot_number, block_number, categ, t_on, t_off, t_warn,
t_end, and error, representing the processed session data.

TODO: Determine slot's t_end based on the next slot's t_on
TODO: Add support for the CLK task (t_warn field): two stimuli are presented in each slot (warning
TORC and click sound). Therefore, the processing should be adapted to handle two stimuli per slot,
and in this case the first stimulus is category N ('neutral') and the second is either R or T.
Should I create two instances of StimulusCategory for the two tasks PDT and CLK to implement two
distinct mappings ?
TODO: Take into account the BEHAVIOR,SHOCKON convention (no space) to perform the split by comas and
spaces and give a single label "SHOCK" in the enum class.
TODO: Assign category "0" to the first reference stimulus in each block ?

"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import numpy as np

from core.processors.base_processor import Processor


# TODO: Replace by EventType from core.attributes.exp_factors
class EventType(str, Enum):
    """
    Enum class for the valid types of events in the raw data.

    - TRIALSTART: Marks the start of a block.
    - PRESTIM: Marks the pre-stimulus silence period within a slot.
    - STIM: Marks the onset of a stimulus within a slot.
    - POSTSTIM: Marks the post-stimulus silence period within a slot.
    - BEHAVIOR: Represents behavioral responses.
    - SHOCKON: Marks the occurrence of a shock on error trials.
    - TRIALSTOP: Marks the end of a block.
    """

    TRIALSTART = "TRIALSTART"
    PRESTIM = "PreStimSilence"
    STIM = "Stim"
    POSTSTIM = "PostStimSilence"
    SHOCK = "BEHAVIOR,SHOCKON"
    TRIALSTOP = "TRIALSTOP"


# TODO: Replace by Category from core.attributes.exp_factors
class StimulusCategory:
    """Represent stimulus category.

    Class Attributes
    ----------------
    CATEGORY_MAP : dict
        Mapping from full labels to aliases.
    UNKNOWN_LABEL : str
        Default label for unknown categories.

    Methods
    -------
    from_string(category: str) -> str
    """

    CATEGORY_MAP = {"Target": "T", "Reference": "R"}
    UNKNOWN_LABEL = "UNKNOWN"

    @classmethod
    def from_string(cls, category: str) -> str:
        """
        Map the full category name to its alias.

        Parameters
        ----------
        category : str
            Full category name (e.g., "Target" or "Reference").

        Returns
        -------
        str
            Alias for the stimulus category (e.g., "T" or "R").
            If the category is not recognized, return 'UNKNOWN'.
        """
        return cls.CATEGORY_MAP.get(category, cls.UNKNOWN_LABEL)


@dataclass
class EventManager:
    """
    Represents an individual event in the raw data and extracts its components.

    Attributes
    ----------
    event_number : int
        Event number in the raw data.
    block : int
        Block number in which the event occurred.
    t_start : float
        Start time of the event (in seconds) relative to the start of the session.
    t_end : float
        Stop time of the event (in seconds) relative to the start of the session.
    description : str
        Description of the event in the raw data (from the field 'Event' in the raw data).
    event_type : {"TRIALSTART", "PreStimSilence", "Stim", "PostStimSilence", "BEHAVIOR",
    "SHOCKON", "TRIALSTOP"}
        Type of event, among the possible values. See the `EventTypes` enum.
    stimulus : Optional[str]
        Stimulus identity (if applicable). Examples: 'TORC_448_06_v501', '2000', '8000'...
    category : Optional[str]
        Stimulus category (if applicable). Examples: 'Reference', 'Target'...
    intensity : Optional[str]
        Sound intensity (if applicable). Examples: '0dB', '10dB'...

    Methods
    -------
    parse_description()
    is_type(event_type: str) -> bool

    Examples
    --------

    Create an `EventManager` object and access its attributes:

    >>> event = EventManager(1, 1, 0.0, 0.4, "'PreStimSilence , TORC_448_06_v501 , Reference'", None)
    >>> event.event_type
    <EventType.PRESTIM: 'PreStimSilence'>
    >>> event.stimulus
    'TORC_448_06_v501'
    >>> event.category
    'Reference'
    >>> event.intensity
    None

    """

    event_number: int
    block: int
    t_start: float
    t_end: float
    description: str
    event_type: EventType = field(init=False)
    stimulus: Optional[str] = field(init=False)
    category: Optional[str] = field(init=False)
    intensity: Optional[str] = field(init=False)

    def __post_init__(self):
        self.parse_description()

    def parse_description(self):
        """
        Split the description string of an event into its components.

        See Also
        --------
        `split`: Split a string into a list of substrings, using a delimiter.
        `strip`: Remove leading and trailing whitespaces from a string.
        """
        parts = [p.strip() for p in self.description.split(",")]
        self.event_type = EventType(parts[0])  # validate event type
        self.stimulus = parts[1] if len(parts) > 1 else ""
        self.category = StimulusCategory.from_string(parts[2]) if len(parts) > 2 else ""
        self.intensity = parts[3] if len(parts) > 3 else ""

    def is_type(self, event_type: EventType) -> bool:
        """Check if the event is of a specific type."""
        return self.event_type == EventType(event_type)


@dataclass
class SlotManager:
    """
    Represents the information about one slot in the session.

    One slot corresponds to a segment of a block where specific events related to stimulus
    presentation occur.

    Attributes
    ----------
    slot_number : int
        Slot number within the bock to which it belongs.
    block_number : int
        Block number in which the slot occurred.
    categ : Optional[str]
        Stimulus identity.
    t_on : Optional[float]
        Start time of the slot (onset of the stimulus).
    t_off : Optional[float]
        End time of the slot.
    t_warn : Optional[float]
        Time of the warning sound in the CLK task.
    t_end : Optional[float]
        End time of the slot.
    error : bool
        Error status of the slot. True if the shock was delivered.

    Examples
    --------

    Create a `SlotManager` object and access its attributes:

    >>> slot = SlotManager(1, 1)
    >>> slot.slot_number
    1
    >>> slot.block_number
    1

    """

    slot_number: int
    block_number: int
    categ: Optional[str] = None
    t_on: Optional[float] = None
    t_off: Optional[float] = None
    t_warn: Optional[float] = None
    t_end: Optional[float] = None
    error: bool = False


@dataclass
class BlockManager:
    """
    Manages the information and lifecycle of a block within the session.

    One block consists of multiple slots, each containing stimulus presentation events.

    Attributes
    ----------
    block_number : int
        Block number in the raw data.
    slots : List[SlotManager]
        Slots belonging to the block.
    t_start : float
        Start time of the block (in seconds) relative to the start of the session.
    t_end : float
        Stop time of the block (in seconds) relative to the start of the session. Initialized to
        None and set when the block is closed.

    Methods
    -------
    start_slot()
    close_slot()
    process_event(event: EventManager)

    Examples
    --------

    Create a `BlockManager` object and access its attributes:

    >>> block = BlockManager(1, 0.0, 0.4)
    >>> block.block_number
    1
    >>> block.t_start
    0.0
    >>> block.t_end
    0.4

    """

    block_number: int
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    slots: list[SlotManager] = field(default_factory=list)
    current_slot: Optional[SlotManager] = None
    current_event: Optional[EventManager] = None

    def start_slot(self):
        """Start a new slot in the block."""
        slot_number = len(self.slots) + 1  # increment slot number (start at 1)
        self.current_slot = SlotManager(slot_number, self.block_number)

    def close_slot(self):
        """Close the current slot in the block."""
        self.slots.append(self.current_slot)
        self.current_slot = None

    def process_event(self, event: EventManager):
        """Process events and assign them to the current slot."""
        if event.is_type(EventType.PRESTIM):
            self.start_slot()
        elif event.is_type(EventType.STIM):
            setattr(self.current_slot, "categ", event.stimulus)
            setattr(self.current_slot, "t_on", event.t_start)
        elif event.is_type(EventType.POSTSTIM):
            setattr(self.current_slot, "t_off", event.t_end)
            self.close_slot()
        elif event.is_type(EventType.SHOCK):
            setattr(self.current_slot, "error", True)


class SessionParser(Processor):
    """
    Extract trials information from raw data relative to the events in one session.

    Class Attributes
    ----------------

    Configuration Attributes
    ------------------------

    Processing Arguments
    --------------------
    events : List[Tuple[int, int, float, float, str]]
        Raw event data, each specifying event number, block, start time, end time, and description.
        .. _events:

    Returns
    -------
    output : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray]
        Arrays containing information for each slot:

        - slot_number: Number of the slot within the block.
        - block_number: Number of the block to which the slot belongs.
        - categ: Nature of the stimulus presented in the slot.
        - t_on: Onset time of the stimulus presentation within the slot.
        - t_off: Offset time of the stimulus presentation within the slot.
        - t_warn: Onset of the warning sound (in the CLK task only).
        - t_end: End time of the slot.
        - error: Behavioral choice of the slot, if Target (True if the shock was delivered).

        Length of each array: ``(n_slots,)``

    Methods
    -------

    Examples
    --------

    >>> events = [
    ...     (1, 1, 0.0, 0.0, "TRIALSTART"),
    ...     (2, 1, 0.0, 0.4, "'PreStimSilence , TORC_448_06_v501 , Reference'"),
    ...     (3, 1, 0.4, 0.8, "'Stim , TORC_448_06_v501 , Reference'"),
    ...     (4, 1, 0.8, 1.2, "'PostStimSilence , TORC_448_06_v501 , Reference'"),
    ...     (5, 1, 1.2, 1.6, "TRIALSTOP"),
    ...     (6, 2, 1.6, 1.6, "TRIALSTART"),
    ...     (7, 2, 1.6, 2.0, "'PreStimSilence , TORC_448_06_v501 , Reference'"),
    ...     (8, 2, 2.0, 2.4, "'Stim , TORC_448_06_v501 , Reference'"),
    ...     (9, 2, 2.4, 2.8, "'PostStimSilence , TORC_448_06_v501 , Reference'"),
    ...     (10, 2, 2.8, 3.2, "TRIALSTOP"),
    ... ]
    >>> parser = SessionParser()
    >>> results = parser.process(events=events) # all output arrays in a tuple
    >>> slot, block, categ, t_on, t_off, t_warn, t_end, error = results # unpack

    See Also
    --------
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    events: list[EventManager]
    blocks: list[BlockManager]
    current_block: Optional[BlockManager]

    def __init__(self):
        super().__init__()

    def _process(self, **input_data):
        """Implement the template method called in the base class `process` method."""
        events = input_data["events"]
        self.reset(events)
        return self.aggregate_results()

    def reset(self, events):
        """Reset the processor with new events."""
        self.events = [EventManager(*event) for event in events]
        self.blocks = []
        self.current_block = None

    # --- Processing Methods -----------------------------------------------------------------------

    def start_block(self, t_start):
        """Initialize a new block."""
        block_number = len(self.blocks) + 1  # increment block number (start at 1)
        self.current_block = BlockManager(block_number)
        setattr(self.current_block, "t_start", t_start)

    def close_block(self, t_end):
        """Set the end time of the block."""
        setattr(self.current_block, "t_end", t_end)
        self.blocks.append(self.current_block)
        self.current_block = None

    def process_event(self, event: EventManager):
        """Process events and assign them to the current block."""
        if event.is_type(EventType.TRIALSTART):
            self.start_block(event.t_start)
        elif event.is_type(EventType.TRIALSTOP):
            self.close_block(event.t_end)
        elif self.current_block is not None:
            self.current_block.process_event(event)

    def aggregate_results(self):
        """Aggregate results into output arrays."""
        all_slots = [slot for block in self.blocks for slot in block.slots]
        slot_number = np.array([slot.slot_number for slot in all_slots])
        block_number = np.array([slot.block_number for slot in all_slots])
        categ = np.array([slot.categ for slot in all_slots])
        t_on = np.array([slot.t_on for slot in all_slots])
        t_off = np.array([slot.t_off for slot in all_slots])
        t_warn = np.array([slot.t_warn for slot in all_slots])
        t_end = np.array([slot.t_end for slot in all_slots])
        error = np.array([slot.error for slot in all_slots])
        return slot_number, block_number, categ, t_on, t_off, t_warn, t_end, error
