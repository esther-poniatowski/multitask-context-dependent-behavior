"""
:mod:`core.parse.initial` [module]

Initial code in previous version of the package.

See Also
--------


"""

from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt


# ------ EVENTSinfo - TRIALSinfo - Information about all trials ------


def Extract_Events_in_Session(session, PATHS):
    """Extracts events in *one* session.
    The events are gathered in a CSV file exported from the .m files (MATLAB script).
    Fields (columns)        Example
    ----------------        -------
        TrialNum (int)      1
        Event (str)         'PreStimSilence , TORC_448_06_v501 , Reference'
        StartTime (float)   0.40
        StopTime (float)    1.65
    For the field 'Event', each string has to be split into its different elements, i.e. :
        * the type of event (e.g. 'TRIALSTART', 'PreStimSilence', 'Stim'...),
        * the precise stimulus identity ('TORC_448_06_v501', '2000'...)
        * the type (category) of the stimulus (Reference'/'Target').
    PROCEDURE : s.split(' , ')
    WARNING : The relevant events are labelled by 'Stim'. However, it is necessary to extract *all* the events (even irrelevant ones : 'TRIALSTART', 'PreStimSilence', 'PostStimSilence', 'TRIALSTOP'), because it is not possible to directly merge the information from different lines of the CSV file corresponding to a single trial (since each trial is associated to multiple events, described on several lines).
    The task of gathering all the (relevant) events of the same trial is performed by another function.
    ------------------
    Inputs :
        session (str)   Name of the session.
    Outputs:
        Events (dict)   All events in the session.
                        Keys indicate different types of information about an event (see below).
                        Values are lists, each index corresponds to one *event*.
                        Indices match across lists for different keys.
        Keys                    Values
        ----                    ------
        TrialNum (list of int)  Trial number within the session.
        Event (list of list)    Information about the event.
                                    event[0]            TRIALSTART / PreStimSilence / Stim / PostStimSilence / BEHAVIOR,SHOCKON / TRIALSTOP
                                    event[1] (optional) TORC_424_08_v501 / 2000 / 8000 ...
                                    event[2] (optional) Reference / Target
                                    event[3] (optional) 0dB
        StartTime (list of float)   Starting time of the event (in sec).
        StopTime (list of float)    End time of the event (in sec).
    """
    Events = Open_CSVFile_ExptEvents(session, PATHS)
    Events["Event"] = [s.split(" , ") for s in Events["Event"]]
    return Events


def Gathers_Events_in_Trials(session, Events, TrialTrev=Reverse_Dictionary(TRIAL_TYPES)):
    """Gathers events according to trials in one session.
    NOTE : Only relevant events (stimuli) are retained for each trial.
    Pre-stimulus and post-stimulus durations can be recovered from the time intervals between the different stimuli.
    ------------------
    Inputs :
        session (str)       Name of the session.
        Events (dict)       All events in the session (see Extract_Events_in_Session()).
                            Keys : 'TrialNum', 'Event', 'StartTime', 'StopTime'.
        TrialTrev (dict)    Dictionary associating trial types as tuples to their index in TRIAL_TYPES.
    Output :
        Trials (dict)       Trials of the session.
                            Keys indicate different types of information about a trial.
                            Values are lists, each index corresponds to one *trial*.
                            Indices match across lists for different keys.
        Keys                        Values
        --------------------        -------
        Session (list of str)       Common session of all the trials (repeated).
        TrialNum (list of int)      Trial number within the session (>= 1)
        TrialType (int)             Code indicating the sequence of stimuli presentations.
                                    Convention (See TRIAL_TYPES defined in GLOBAL_VARIABLES) :
                                        * Numbers from 1 to 6 indicate the position of the target.
                                        * 0 indicates a catch trial.
        TargType (list)             Target type (tone frequency or click rate, in Hz).
                                    None if catch trial.
        StimTimes (list of lists)   Start and end times (in sec) of all the stimuli within each trial, relative to the beginning of the trial.
                                    Each element is a sub-list of at most 7 elements depending on the type of the trial.
        Error (list of bool)        Success or failure (behavior of the animal) during the shock window.
        Duration (list of float)    Duration of the trials (in sec).
    """
    # Initialize dictionary
    Trials = {"TrialNum": Unique(Events["TrialNum"])}  # all trials in the session
    Ntrials = len(Trials["TrialNum"])
    Trials["Session"] = [session] * Ntrials  # same session for all trials
    Trials["Error"] = [False] * Ntrials  # success by default for all trials
    Trials["TargType"] = [None] * Ntrials  # no target by default for all trials
    Trials["StimTimes"] = [
        [] for _ in range(Ntrials)
    ]  # empty list to be appended with times of stimuli
    Trials["Stimulus"] = [
        [] for _ in range(Ntrials)
    ]  # empty list to be appended with types of stimuli
    Trials["Duration"] = [0] * Ntrials  # initialize to 0 by default
    # WARNING : do not use the syntax [[]]*Ntrials because lists would not be independent
    # 1/ Sweep events to fill StimTimes, TargType, Error
    for num, event, t_start, t_end in zip(
        Events["TrialNum"], Events["Event"], Events["StartTime"], Events["StopTime"]
    ):
        i = num - 1  # index of the trial, decremented by 1 to start at 0 instead of 1
        event_type = event[
            0
        ]  # TRIALSTART/PreStimSilence/Stim/PostStimSilence/TRIALSTOP/BEHAVIOR,SHOCKON
        if event_type == "Stim":  # filter only Reference and Target events
            sound_type = event[1]  # nature of the sound : TORC, pure tone rate, click rate...
            stim_type = event[2]  # target/reference
            Trials["StimTimes"][i].append((t_start, t_end))  # append the time boundaries
            if stim_type == "Target":
                Trials["Stimulus"][i].append("T")  # append a target
                if (":" not in sound_type) and (
                    "TORC" not in sound_type
                ):  # avoid some complex strings in CCH tasks and pre-click noise in CLK
                    sound_type = float(
                        sound_type.replace("[", "").replace("]", "")
                    )  # e.g. '[20]' for CLK
                Trials["TargType"][i] = sound_type
            else:  # append a reference
                Trials["Stimulus"][i].append("R")
        elif event_type == "TRIALSTOP":
            Trials["Duration"][i] = t_end  # duration of the trial = end of the TRIALSTOP event
        elif event_type == "BEHAVIOR,SHOCKON":
            Trials["Error"][i] = True
    # 2/ For CLK, fuse TORC and Click times in StimTimes within each trial
    task = Session_Features(session)[2]
    if task == "CLK":
        for i in range(Ntrials):
            Trials["StimTimes"][i] = [
                (t1[0], t2[1])
                for t1, t2 in zip(Trials["StimTimes"][i][::2], Trials["StimTimes"][i][1::2])
            ]  # by pairs of 2 tuples TROC/Click : (t0_TORC, t1_TORC), (t0_clk, t1_clk) -> (t0_TORC, t1_clk)
            Trials["Stimulus"][i] = Trials["Stimulus"][i][
                ::2
            ]  # keep one element out of 2 : [R_TORC, R_clk, T_TORC, T_clk] -> [R, T]
    # 3/ Determine TrialType based on the nature of stimuli in each trial
    StimSeqs = [
        tuple([x == "T" for x in Stimuli]) for Stimuli in Trials["Stimulus"]
    ]  # convert to boolean code
    Trials["TrialType"] = [
        TrialTrev[stim_seq] if stim_seq in TrialTrev else None for stim_seq in StimSeqs
    ]
    # 4/ Set the first reference
    for i in range(Ntrials):
        Trials["Stimulus"][i][0] = "0"
    return Trials


def Find_Durations(StimTimes, tstop):
    """Durations of the stimulus, pre-stimulus, post-stimulus periods of the events in one trial.
    ------------------
    Inputs :
        StimTimes (list of tuples)  Start and end times (t0,t1) of each stimulus in each trial.
        tstop (float)               Duration of the trial.
    Outputs :
        DStim (list of float)       Duration of the stimulus in each event.
        DPreStim                    Duration of the epoch preceding the stimulus.
        DPostStim                   Duration of the epoch following the stimulus.
    """
    DStim, DPreStim, DPostStim = [], [], []
    # First stimulus
    t0, t1 = StimTimes[0]
    DPreStim.append(np.round(t0, decimals=3))  # first prestimulus period
    DStim.append(np.round(t1 - t0, decimals=3))
    tend = t1
    # Following stimuli
    for t0, t1 in StimTimes[1:]:
        DPostStim.append(np.round(t0 - tend, decimals=3))  # for the previous stimulus
        DPreStim.append(DPostStim[-1])  # for the current stimulus
        DStim.append(np.round(t1 - t0, decimals=3))
        tend = t1
    # Last stimulus
    DPostStim.append(np.round(tstop - tend, decimals=3))
    return DStim, DPreStim, DPostStim


def Find_StimTypes(ttype, TRIAL_TYPES=TRIAL_TYPES):
    """Types of all the stimuli in one trial.
    ------------------
    Inputs :
        ttypes (int)    Code indicating the type of trial (see TRIAL_TYPEs defined in GLOBAL_VARIABLES).
    Outputs :
        Stimuli (list of str)  Sequence of '0', 'R' and 'T' corresponding to the stimuli in the trial.
    """
    return ["0"] + ["R" if stim == 0 else "T" for stim in TRIAL_TYPES[ttype][1:]]


def Build_EVENTSinfo_TRIALSinfo(UNITSinfo, PATHS, D=D):
    """Events of the experiment and Trials of the experiment.
    NOTE : The durations of the stimulus, pre-stimulus and post-stimulus period in EVENTSinfo can be used to select events for the final analysis.
    ------------------
    Input :
        PATHS (dict)                    Paths of the raw data.
                                        Keys (str)      Sessions' labels (e.g. 'avo052a04_p_PTD_1')
                                        Values (str)    Path of the sessions' data.
    Output :
        EVENTSinfo (dataframe)
            Index                       Multilevel index : (Session (str), EventNum (int))
            -------                     NOTE : EventNum has no meaning in the experiment, it is only used in EVENTSinfo to get unique indices.
            Columns
            -------
            Session (str)               Session in which this event appeared.
            TrialNum (int)              Index of the trial in which this event appeared.
            Position (int)              Position of the event within the trial (0, ..., 6).
            StimTimes (tuple of floats) (t0, t1), start and end times of stimulus (in sec), relative to the beginning of the trial.
            DStim (float)               Duration of the stimulus (in sec).
            DPre (float)                Duration of the pre-stimulus period (in sec).
            DPost (float)               Duration of the post-stimulus period (in sec).
            Stimulus (str)              '0' (First Reference) / 'R' (References in positions 1-6) / 'T' (Target)
            Valid (bool)                If True, the durations of the pre-stimulus, stimuuls and post-stimulus periods are sufficient to (potentially) include this event in the analyses.
                                        Here, initialized to False for all events.
                                        Updated in the function PreSelect_Events().
            Index (int)                 Index of the trial in the third dimension of the (future) *firing rate matrices* of PSTH.
                                        None if the event is excluded from the analyses.
                                        Here, initialized to None by default for all events.
                                        Updated in the function Select_Trials().
       TRIALSinfo (dataframe)
            Index                       Multilevel index : (Session (str), TrialNum (int))
            -------                     Session in which the trial belongs.
                                        Index of the trial within its session.
            Columns
            -------
            Session (str)               Session.
            TrialNum (int)              Index of the trial within its session.
            TrialType (int)             Code indicating the stimulus sequence.
                                        Convention (See TRIAL_TYPES) :
                                            * Numbers from 1 to 6 indicate the postiion of the target.
                                            * 0 indicates a catch trial.
            TargType (float)            Nature of the target type (tone frequency or click rate, in Hz). None if catch trial.
            Error (bool)                Success or failure during the shock window.
    """
    # Initialize dictionaries
    KEYS_TRIALSinfo = ["Session", "TrialNum", "TrialType", "TargType", "Error", "Duration"]
    KEYS_EVENTSinfo = [
        "Session",
        "TrialNum",
        "Position",
        "StimTimes",
        "DStim",
        "DPre",
        "DPost",
        "Stimulus",
        "Valid",
        "Index",
        "EventNum",
    ]
    TRIALSinfo = {key: [] for key in KEYS_TRIALSinfo}
    EVENTSinfo = {key: [] for key in KEYS_EVENTSinfo}
    A = (
        UNITSinfo[["Site", "Area"]].drop_duplicates().set_index("Site")
    )  # to recover the sessions in which the events were recorded, and then the minimal durations required for the valid events (columns 'Area' and 'Site' are in one to one mapping)
    # 1/ Fill TRIALSinfo by sweeping *sessions*
    Nsess = len(PATHS)
    for s, session in enumerate(PATHS):
        print(f"Session {s}/{Nsess} | {session}")
        Events = Extract_Events_in_Session(session, PATHS)
        Trials = Gathers_Events_in_Trials(session, Events)
        # Keys : 'Session', 'TrialNum', 'TrialType', 'TargType', 'StimTimes', 'Error', 'Duration'.
        # Values : Lists for each key, each containing the information of all the trials of the session.
        for key in KEYS_TRIALSinfo:  # WARNING : KEYS_TRIALSinfo and Trials have common keys
            TRIALSinfo[key] += Trials[key]  # merge the list of all trials within the session
        # 2/ Fill EVENTSinfo by sweeping *trials*
        for num, ttype, StimTimes, tstop in zip(
            Trials["TrialNum"], Trials["TrialType"], Trials["StimTimes"], Trials["Duration"]
        ):  # one iteration = one trial
            NStim = len(StimTimes)
            EVENTSinfo["Session"] += [
                session
            ] * NStim  # same session for all the events in the trial
            EVENTSinfo["TrialNum"] += [num] * NStim  # same trial for all the events in the trial
            EVENTSinfo["Position"] += [pos for pos in range(NStim)]
            EVENTSinfo["StimTimes"] += StimTimes
            DStim, DPreStim, DPostStim = Find_Durations(
                StimTimes, tstop
            )  # durations of all the stimuli within the trial
            EVENTSinfo["DStim"] += DStim
            EVENTSinfo["DPre"] += DPreStim
            EVENTSinfo["DPost"] += DPostStim
            EVENTSinfo["Stimulus"] += Find_StimTypes(ttype)
    Nevents = len(EVENTSinfo["StimTimes"])
    EVENTSinfo["Valid"] = [False] * Nevents  # not valid by default
    EVENTSinfo["Index"] = [None] * Nevents  # not selected by default
    EVENTSinfo["EventNum"] = [i for i in range(Nevents)]  # for the index
    EVENTSinfo, TRIALSinfo = pd.DataFrame(EVENTSinfo), pd.DataFrame(TRIALSinfo)
    EVENTSinfo.set_index(["Session", "EventNum"], inplace=True, drop=False)
    TRIALSinfo.set_index(["Session", "TrialNum"], inplace=True, drop=False)
    return EVENTSinfo, TRIALSinfo


# ------ SESSIONSinfo - Information about all sessions ------


def Update_Contexts(SESSIONSinfo, SESSIONS, verbose=False):
    """Updates the context with pre-passive and post-passive sessions of a *common* site.
    NOTE : Common sessions recorded on the same site on the same recording day are found in the dictionary SESSIONS.
    That is why this step is not performed during the sweeping of sessions in Build_SESSIONSinfo :
    this sweeping does not take into account the link between different sessions of the same site.
    WARNING : In some cases, several active sessions are recorded on the same site,
    which leads to a sequence [p, a, p, ..., p, a, p, ...].
    In this case, pre-passive sessions are assigned also before the second active one,
    only if there is at least two passive sessions between the two active sessions :
    this allows to ascribe one post-passive after the first active
    and one pre-passive before the second active.
    ------------------
    Input:
        SESSIONS (dict)             Sessions of all sites.
                                    Keys (str)              Site label (e.g. 'ath011b')
                                    Values (list of str)    List of sessions (e.g. ['ath011b03_p_PTD', 'ath011b04_a_PTD', 'ath011b05_p_PTD'])
        SESSIONSinfo (dataframe)    Version of SESSIONSinfo already under the form of a dataframe.
    Output:
        SESSIONSinfo (dataframe)    Updated version in the column 'Context'.
    """
    Nsites = len(SESSIONS)
    for s, SessionsOfSite in enumerate(SESSIONS.values()):  # common sessions of each site
        if verbose:
            print(f"Site {s}/{Nsites}")
        # Order in the temporal succession of recordings
        Recordings = [SESSIONSinfo.loc[session, "Recording"] for session in SessionsOfSite]
        I = Sorted_Indices(Recordings)
        SessionsOfSite = [SessionsOfSite[i] for i in I]
        Recordings = [Recordings[i] for i in I]
        Contexts = [SESSIONSinfo.loc[session, "Context"] for session in SessionsOfSite]
        # Update the context in SESSIONSinfo
        I_act = Indices_Elements(Contexts, "a")
        if len(I_act) > 0:  # at least one active session
            # deal with the first active session
            i_a = I_act[0]  # index of the first active session
            for i in range(i_a):  # for all passive sessions before the first active
                SESSIONSinfo.loc[SessionsOfSite[i], "Context"] = "p-pre"
            for i in range(i_a + 1, len(Contexts)):  # for all sessions after the first active
                if Contexts[i] != "a":  # if passive session
                    SESSIONSinfo.loc[SessionsOfSite[i], "Context"] = "p-post"
        if len(I_act) > 1:  # consider next active sessions
            for i_a0, i_a1 in zip(I_act[:-1], I_act[1:]):  # pairs of active sessions
                if i_a1 - i_a0 >= 2:  # at least two passive sessions between them
                    SESSIONSinfo.loc[SessionsOfSite[i_a1 - 1], "Context"] = (
                        "p-pre"  # ascribe pre-passive to the one before the second active
                    )
    return SESSIONSinfo


def Build_SESSIONSinfo(TRIALSinfo, SESSIONS):
    """Sessions of the experiment.
    NOTE : The columns 'Area', 'Task', 'Context', 'NTarg' are used for selection.
    ------------------
    Output: SESSIONSinfo (dataframe)
        Index               Name of the session.
        -------
        Columns
        -------
        Session (str)       Name of the session.
        Site (str)          Recording site (e.g. 'avo052a').
        Recording (int)     Recording number at this site (e.g. 04).
        Task (str)          'PTD'/'CCH'
        Context (str)       'p'/'p-pre'/'a'/'p-post' ('Passive'/'Pre-Passive'/'Active'/'Post-Passive')
                                NOTE : 'p' indicates that this session belongs to a recording day without active sessions.
        NTrials (int)       Number of trials in this session.
        NT (int)            Number of VALID targets in this session.
                                Here, initialized to 0 by default for all sessions.
                                Updated in the function Select_Units().
        NR (int)            Idem for VALID references in positions 1-6.
        N0 (int)            Idem for VALID first reference.
        Valid (bool)        If True, the session owns at least the required number of events of each type.
                                Here, initialized to False by default for all sessions.
    """
    # Initialize dictionary
    KEYS = ["Session", "Site", "Recording", "Task", "Context", "NTrials"]
    SESSIONSinfo = {key: [] for key in KEYS}
    # Fill by sweeping sessions
    Sessions = Unique(TRIALSinfo["Session"])
    Nsess = len(Sessions)
    for s, session in enumerate(Sessions):
        print(f"Session {s}/{Nsess} | {session}")
        site, rec, task, ctx = Session_Features(session)
        NTrials = len(TRIALSinfo.loc[session, "TrialNum"])
        for key, value in zip(
            KEYS, [session, site, rec, task, ctx, NTrials]
        ):  # WARNING : ensure the order of the value matches that in KEYS
            SESSIONSinfo[key].append(value)
    SESSIONSinfo["NT"] = [0] * Nsess
    SESSIONSinfo["NR"] = [0] * Nsess
    SESSIONSinfo["N0"] = [0] * Nsess
    SESSIONSinfo["Valid"] = [False] * Nsess
    SESSIONSinfo = pd.DataFrame(SESSIONSinfo, index=SESSIONSinfo["Session"])
    print("Update Contexts")
    SESSIONSinfo = Update_Contexts(SESSIONSinfo, SESSIONS)
    return SESSIONSinfo
