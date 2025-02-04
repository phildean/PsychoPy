#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on February 04, 2025, at 16:36
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'n-back'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': 'subj-',
    'session': 'sess-01',
    'backN': '2',
    'probability': '0.3',
    'blocks': '5',
    'trials': '50',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'Data/%s_%s_%s_%s' % (expInfo['participant'], expInfo['session'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\pss1pd\\OneDrive - University of Surrey\\Documents\\Programs_Scripts\\PsychoPy\\n-BackTask\\n-back_task_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1,-1,-1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('instructions_key') is None:
        # initialise instructions_key
        instructions_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instructions_key',
        )
    if deviceManager.getDevice('keyBlockStart') is None:
        # initialise keyBlockStart
        keyBlockStart = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyBlockStart',
        )
    if deviceManager.getDevice('trial_resp') is None:
        # initialise trial_resp
        trial_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='trial_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Instructions" ---
    # Run 'Begin Experiment' code from codeInstructions
    ### Experimental Setup
    # Setup Variables
    blockNum = 1
    
    ##### Uncomment the below for EEG Markers
    
    #### Setting up port to send EEG triggers to
    #import serial
    #port = serial.Serial('COM3')
    
    ## timeout 1 second (give 1 second to the board to initialize port)
    #port.timeout = 1
    
    ## Send blank to port to set it all to zero. 
    #port.write([0x0])
    
    #### Markers
    #mrkBlockStart = 10
    #mrkBlockEnd = 11
    #mrkNonTarget = 20
    #mrkTarget = 21
    #mrkResponse = 5
    instructions = visual.TextStim(win=win, name='instructions',
        text="The n-back task presents stimuli sequentially and \n requires you to decide if the current stimulus is \n the same as one presented n-numbers previous (or back).\n"
    + "\n"
    + "For example, the 2-back you will indicate if the current letter \n matches the letter presented 2 places previous\n"
    + "(e.g.  A H G A B G B shows that 'A' and 'G' have matches in this sequence).\n"
    + "\n\n"
    + f"This is a {expInfo['backN']}-back task.\n"
    + f"Press SPACE if the current stimulus is the same as {expInfo['backN']} places previous.\n"
    + "\n\n"
    + "Press SPACE to continue",
        font='Arial',
        pos=(0, 0), draggable=False, height=1, wrapWidth=45, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    instructions_key = keyboard.Keyboard(deviceName='instructions_key')
    # Set experiment start values for variable component stimuli
    stimuli = np.array(['A','B','C','D','E','F','G','H'])
    stimuliContainer = []
    
    # --- Initialize components for Routine "blockStart" ---
    textBlockStart1 = visual.TextStim(win=win, name='textBlockStart1',
        text='Block:',
        font='Arial',
        pos=(0, 2), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    textBlockStart2 = visual.TextStim(win=win, name='textBlockStart2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    textBlockStart3 = visual.TextStim(win=win, name='textBlockStart3',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -2), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    keyBlockStart = keyboard.Keyboard(deviceName='keyBlockStart')
    
    # --- Initialize components for Routine "trial" ---
    # Run 'Begin Experiment' code from trial_code
     # Initialise an empty array to keep track of 
     # previous stimuli.
     
    prevN = np.array([])
    trial_stimulus = visual.TextStim(win=win, name='trial_stimulus',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=2.5, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    trial_resp = keyboard.Keyboard(deviceName='trial_resp')
    
    # --- Initialize components for Routine "blockEnd" ---
    textBlockEnd = visual.TextStim(win=win, name='textBlockEnd',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Confidencerating" ---
    text_confidencetask = visual.TextStim(win=win, name='text_confidencetask',
        text='Please select your confidence level \nin your performance from 0 to 100 %',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=1.0, wrapWidth=25.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    slider_confidence = visual.Slider(win=win, name='slider_confidence',
        startValue=None, size=(25, 1), pos=(0, -3), units=win.units,
        labels=["0", "25", "50", "75", "100"], ticks=(1, 2, 3, 4, 5), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='white', colorSpace='rgb',
        font='Open Sans', labelHeight=1.0,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Instructions" ---
    # create an object to store info about Routine Instructions
    Instructions = data.Routine(
        name='Instructions',
        components=[instructions, instructions_key],
    )
    Instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instructions_key
    instructions_key.keys = []
    instructions_key.rt = []
    _instructions_key_allKeys = []
    # store start times for Instructions
    Instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instructions.tStart = globalClock.getTime(format='float')
    Instructions.status = STARTED
    thisExp.addData('Instructions.started', Instructions.tStart)
    Instructions.maxDuration = None
    # keep track of which components have finished
    InstructionsComponents = Instructions.components
    for thisComponent in Instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions" ---
    Instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions* updates
        
        # if instructions is starting this frame...
        if instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions.frameNStart = frameN  # exact frame index
            instructions.tStart = t  # local t and not account for scr refresh
            instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            instructions.status = STARTED
            instructions.setAutoDraw(True)
        
        # if instructions is active this frame...
        if instructions.status == STARTED:
            # update params
            pass
        
        # if instructions is stopping this frame...
        if instructions.status == STARTED:
            if bool(False):
                # keep track of stop time/frame for later
                instructions.tStop = t  # not accounting for scr refresh
                instructions.tStopRefresh = tThisFlipGlobal  # on global time
                instructions.frameNStop = frameN  # exact frame index
                # update status
                instructions.status = FINISHED
                instructions.setAutoDraw(False)
        
        # *instructions_key* updates
        
        # if instructions_key is starting this frame...
        if instructions_key.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_key.frameNStart = frameN  # exact frame index
            instructions_key.tStart = t  # local t and not account for scr refresh
            instructions_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_key, 'tStartRefresh')  # time at next scr refresh
            # update status
            instructions_key.status = STARTED
            # keyboard checking is just starting
            instructions_key.clock.reset()  # now t=0
        if instructions_key.status == STARTED:
            theseKeys = instructions_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instructions_key_allKeys.extend(theseKeys)
            if len(_instructions_key_allKeys):
                instructions_key.keys = _instructions_key_allKeys[-1].name  # just the last key pressed
                instructions_key.rt = _instructions_key_allKeys[-1].rt
                instructions_key.duration = _instructions_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions" ---
    for thisComponent in Instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instructions
    Instructions.tStop = globalClock.getTime(format='float')
    Instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instructions.stopped', Instructions.tStop)
    
    thisExp.nextEntry()
    # the Routine "Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=expInfo['blocks'], 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "blockStart" ---
        # create an object to store info about Routine blockStart
        blockStart = data.Routine(
            name='blockStart',
            components=[textBlockStart1, textBlockStart2, textBlockStart3, keyBlockStart],
        )
        blockStart.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeBlockStart
        ##### Uncomment the below for EEG Markers
        
        ## Setup whether an EEG trigger sent or not
        #pulse_started = False
        #pulse_ended = False
        textBlockStart2.setText(blockNum)
        # create starting attributes for keyBlockStart
        keyBlockStart.keys = []
        keyBlockStart.rt = []
        _keyBlockStart_allKeys = []
        # store start times for blockStart
        blockStart.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blockStart.tStart = globalClock.getTime(format='float')
        blockStart.status = STARTED
        thisExp.addData('blockStart.started', blockStart.tStart)
        blockStart.maxDuration = None
        # keep track of which components have finished
        blockStartComponents = blockStart.components
        for thisComponent in blockStart.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blockStart" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        blockStart.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from codeBlockStart
            ##### Uncomment the below for EEG Markers
            
            ## Sending trigger after textBlockStart1 started
            #if textBlockStart1.status == STARTED and not pulse_started:
            #    port.write(mrkBlockStart.to_bytes(length = 1, byteorder = 'little'))
            #    blockpulse_start_time = globalClock.getTime()
            #    pulse_started = True
            
            ## Ending the trigger
            #if pulse_started and not pulse_ended:
            #    if textBlockStart1.status == STARTED:
            #        if globalClock.getTime() - blockpulse_start_time >= 0.1:
            #            port.write([0x00])
            #            pulse_ended = True
            
            # *textBlockStart1* updates
            
            # if textBlockStart1 is starting this frame...
            if textBlockStart1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlockStart1.frameNStart = frameN  # exact frame index
                textBlockStart1.tStart = t  # local t and not account for scr refresh
                textBlockStart1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlockStart1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlockStart1.started')
                # update status
                textBlockStart1.status = STARTED
                textBlockStart1.setAutoDraw(True)
            
            # if textBlockStart1 is active this frame...
            if textBlockStart1.status == STARTED:
                # update params
                pass
            
            # *textBlockStart2* updates
            
            # if textBlockStart2 is starting this frame...
            if textBlockStart2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlockStart2.frameNStart = frameN  # exact frame index
                textBlockStart2.tStart = t  # local t and not account for scr refresh
                textBlockStart2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlockStart2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlockStart2.started')
                # update status
                textBlockStart2.status = STARTED
                textBlockStart2.setAutoDraw(True)
            
            # if textBlockStart2 is active this frame...
            if textBlockStart2.status == STARTED:
                # update params
                pass
            
            # *textBlockStart3* updates
            
            # if textBlockStart3 is starting this frame...
            if textBlockStart3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlockStart3.frameNStart = frameN  # exact frame index
                textBlockStart3.tStart = t  # local t and not account for scr refresh
                textBlockStart3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlockStart3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlockStart3.started')
                # update status
                textBlockStart3.status = STARTED
                textBlockStart3.setAutoDraw(True)
            
            # if textBlockStart3 is active this frame...
            if textBlockStart3.status == STARTED:
                # update params
                pass
            
            # *keyBlockStart* updates
            waitOnFlip = False
            
            # if keyBlockStart is starting this frame...
            if keyBlockStart.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                keyBlockStart.frameNStart = frameN  # exact frame index
                keyBlockStart.tStart = t  # local t and not account for scr refresh
                keyBlockStart.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keyBlockStart, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'keyBlockStart.started')
                # update status
                keyBlockStart.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(keyBlockStart.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(keyBlockStart.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if keyBlockStart.status == STARTED and not waitOnFlip:
                theseKeys = keyBlockStart.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _keyBlockStart_allKeys.extend(theseKeys)
                if len(_keyBlockStart_allKeys):
                    keyBlockStart.keys = _keyBlockStart_allKeys[-1].name  # just the last key pressed
                    keyBlockStart.rt = _keyBlockStart_allKeys[-1].rt
                    keyBlockStart.duration = _keyBlockStart_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blockStart.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blockStart.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blockStart" ---
        for thisComponent in blockStart.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blockStart
        blockStart.tStop = globalClock.getTime(format='float')
        blockStart.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blockStart.stopped', blockStart.tStop)
        # Run 'End Routine' code from codeBlockStart
        ##### Uncomment the below for EEG Markers
        
        ## Sending information to logfile
        #thisExp.addData("marker", mrkBlockStart)
        #thisExp.addData("marker.time", blockpulse_start_time)
        # check responses
        if keyBlockStart.keys in ['', [], None]:  # No response was made
            keyBlockStart.keys = None
        blocks.addData('keyBlockStart.keys',keyBlockStart.keys)
        if keyBlockStart.keys != None:  # we had a response
            blocks.addData('keyBlockStart.rt', keyBlockStart.rt)
            blocks.addData('keyBlockStart.duration', keyBlockStart.duration)
        # the Routine "blockStart" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        loop = data.TrialHandler2(
            name='loop',
            nReps=expInfo['trials'], 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(loop)  # add the loop to the experiment
        thisLoop = loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop.rgb)
        if thisLoop != None:
            for paramName in thisLoop:
                globals()[paramName] = thisLoop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop in loop:
            currentLoop = loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop.rgb)
            if thisLoop != None:
                for paramName in thisLoop:
                    globals()[paramName] = thisLoop[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[trial_stimulus, trial_resp],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from trial_code
            # Setup the "non-target / n-Back" and "target / n-Back" trials
            
            # non-target trial is shown if either:
            #   currentLoop.thisN < int(expInfo['backN']): 
            #        Ensures that during the first N trials, 
            #        no N-back trials occur (since there aren't enough past trials yet).
            #   np.random.uniform() > float(expInfo['probability']): 
            #        random number generated between 0 and 1, if 
            #        this exceeds the probability set in expInfo
            #        a non-N-back trial is chosen.
            
            if (currentLoop.thisN < int(expInfo['backN'])) or (np.random.uniform() > float(expInfo['probability'])):
                # delete previously used stimuli to ensure no nBack
                stimleft = np.delete(stimuli,[np.argwhere(stimuli==i) for i in prevN])
                stimulus = np.random.choice(stimleft)
                thisExp.addData('trial_type', 'none')
                correctResponse = None
            else: # for n-Back trial ("target")
                stimulus = prevN[-int(expInfo['backN'])]
                thisExp.addData('trial_type', 'back')
                correctResponse = 'space'
            
            ## Setup whether an EEG trigger sent or not
            #pulse_started = False
            #pulse_ended = False
            
            
            trial_stimulus.setText(stimulus)
            # create starting attributes for trial_resp
            trial_resp.keys = []
            trial_resp.rt = []
            _trial_resp_allKeys = []
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(loop, data.TrialHandler2) and thisLoop.thisN != loop.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from trial_code
                ##### Uncomment the below for EEG Markers
                
                ## Sending trigger after trial_stimulus started
                #if trial_stimulus.status == STARTED and not pulse_started:
                #    if correctResponse = 'space'
                #    port.write(mrkTarget.to_bytes(length = 1, byteorder = 'little'))
                #    trialpulse_start_time = globalClock.getTime()
                #    pulse_started = True
                #    else: 
                #    port.write(mrkNonTarget.to_bytes(length = 1, byteorder = 'little'))
                #    trialpulse_start_time = globalClock.getTime()
                #    pulse_started = True
                
                #kb = KeyBoard()
                #if kb.getKeys() =='space':
                #    port.write(mrkResponse.to_bytes(length = 1, byteorder = 'little'))
                #    response_start_time = globalClock.getTime()
                
                ## Ending the trigger
                #if pulse_started and not pulse_ended:
                #    if trial_stimulus.status == STARTED:
                #        if globalClock.getTime() - trialpulse_start_time >= 0.1:
                #            port.write([0x00])
                #            pulse_ended = True
                
                # *trial_stimulus* updates
                
                # if trial_stimulus is starting this frame...
                if trial_stimulus.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_stimulus.frameNStart = frameN  # exact frame index
                    trial_stimulus.tStart = t  # local t and not account for scr refresh
                    trial_stimulus.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_stimulus, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_stimulus.started')
                    # update status
                    trial_stimulus.status = STARTED
                    trial_stimulus.setAutoDraw(True)
                
                # if trial_stimulus is active this frame...
                if trial_stimulus.status == STARTED:
                    # update params
                    pass
                
                # if trial_stimulus is stopping this frame...
                if trial_stimulus.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_stimulus.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_stimulus.tStop = t  # not accounting for scr refresh
                        trial_stimulus.tStopRefresh = tThisFlipGlobal  # on global time
                        trial_stimulus.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trial_stimulus.stopped')
                        # update status
                        trial_stimulus.status = FINISHED
                        trial_stimulus.setAutoDraw(False)
                
                # *trial_resp* updates
                waitOnFlip = False
                
                # if trial_resp is starting this frame...
                if trial_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_resp.frameNStart = frameN  # exact frame index
                    trial_resp.tStart = t  # local t and not account for scr refresh
                    trial_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_resp.started')
                    # update status
                    trial_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(trial_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(trial_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if trial_resp is stopping this frame...
                if trial_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_resp.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_resp.tStop = t  # not accounting for scr refresh
                        trial_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        trial_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trial_resp.stopped')
                        # update status
                        trial_resp.status = FINISHED
                        trial_resp.status = FINISHED
                if trial_resp.status == STARTED and not waitOnFlip:
                    theseKeys = trial_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _trial_resp_allKeys.extend(theseKeys)
                    if len(_trial_resp_allKeys):
                        trial_resp.keys = _trial_resp_allKeys[0].name  # just the first key pressed
                        trial_resp.rt = _trial_resp_allKeys[0].rt
                        trial_resp.duration = _trial_resp_allKeys[0].duration
                        # was this correct?
                        if (trial_resp.keys == str(correctResponse)) or (trial_resp.keys == correctResponse):
                            trial_resp.corr = 1
                        else:
                            trial_resp.corr = 0
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # Run 'End Routine' code from trial_code
            # Record the current trial's stimulus (stimulus) in the PsychoPy experiment handler (thisExp).
            thisExp.addData('trial_stimulus.stim', stimulus)
            
            # Add the current stimulus to prevN, which is a list tracking previous stimuli.
            prevN = np.hstack((prevN,stimulus))
            
            # Ensure that prevN only stores the most recent N trials (where N = expInfo['backN']).
            prevN = prevN[-int(expInfo['backN']):]
            
            
            ##### Uncomment the below for EEG Markers
            
            ## Sending information to logfile
            #thisExp.addData("marker", mrkNonTarget)
            #thisExp.addData("marker", mrkTarget)
            #thisExp.addData("marker", mrkResponse)
            #thisExp.addData("marker.time", trialpulse_start_time)
            #thisExp.addData("marker.time", response_start_time)
            # check responses
            if trial_resp.keys in ['', [], None]:  # No response was made
                trial_resp.keys = None
                # was no response the correct answer?!
                if str(correctResponse).lower() == 'none':
                   trial_resp.corr = 1;  # correct non-response
                else:
                   trial_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for loop (TrialHandler)
            loop.addData('trial_resp.keys',trial_resp.keys)
            loop.addData('trial_resp.corr', trial_resp.corr)
            if trial_resp.keys != None:  # we had a response
                loop.addData('trial_resp.rt', trial_resp.rt)
                loop.addData('trial_resp.duration', trial_resp.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            thisExp.nextEntry()
            
        # completed expInfo['trials'] repeats of 'loop'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if loop.trialList in ([], [None], None):
            params = []
        else:
            params = loop.trialList[0].keys()
        # save data for this loop
        loop.saveAsExcel(filename + '.xlsx', sheetName='loop',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "blockEnd" ---
        # create an object to store info about Routine blockEnd
        blockEnd = data.Routine(
            name='blockEnd',
            components=[textBlockEnd],
        )
        blockEnd.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeBlockEnd
        ##### Uncomment the below for EEG Markers
        
        ## Setup whether an EEG trigger sent or not
        #pulse_started = False
        #pulse_ended = False
        # store start times for blockEnd
        blockEnd.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blockEnd.tStart = globalClock.getTime(format='float')
        blockEnd.status = STARTED
        thisExp.addData('blockEnd.started', blockEnd.tStart)
        blockEnd.maxDuration = None
        # keep track of which components have finished
        blockEndComponents = blockEnd.components
        for thisComponent in blockEnd.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blockEnd" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        blockEnd.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from codeBlockEnd
            ##### Uncomment the below for EEG Markers
            
            ## Sending trigger after textBlockEnd started
            #if textBlockEnd.status == STARTED and not pulse_started:
            #    port.write(mrkBlockEnd.to_bytes(length = 1, byteorder = 'little'))
            #    blockendpulse_start_time = globalClock.getTime()
            #    pulse_started = True
            
            ## Ending the trigger
            #if pulse_started and not pulse_ended:
            #    if textBlockEnd.status == STARTED:
            #        if globalClock.getTime() - blockendpulse_start_time >= 0.1:
            #            port.write([0x00])
            #            pulse_ended = True
            
            # *textBlockEnd* updates
            
            # if textBlockEnd is starting this frame...
            if textBlockEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlockEnd.frameNStart = frameN  # exact frame index
                textBlockEnd.tStart = t  # local t and not account for scr refresh
                textBlockEnd.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlockEnd, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlockEnd.started')
                # update status
                textBlockEnd.status = STARTED
                textBlockEnd.setAutoDraw(True)
            
            # if textBlockEnd is active this frame...
            if textBlockEnd.status == STARTED:
                # update params
                pass
            
            # if textBlockEnd is stopping this frame...
            if textBlockEnd.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textBlockEnd.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    textBlockEnd.tStop = t  # not accounting for scr refresh
                    textBlockEnd.tStopRefresh = tThisFlipGlobal  # on global time
                    textBlockEnd.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textBlockEnd.stopped')
                    # update status
                    textBlockEnd.status = FINISHED
                    textBlockEnd.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blockEnd.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blockEnd.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blockEnd" ---
        for thisComponent in blockEnd.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blockEnd
        blockEnd.tStop = globalClock.getTime(format='float')
        blockEnd.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blockEnd.stopped', blockEnd.tStop)
        # Run 'End Routine' code from codeBlockEnd
        ## iterate block numner for block start text
        blockNum = blockNum +1
        
        ##### Uncomment the below for EEG Markers
        
        ## Sending information to logfile
        #thisExp.addData("marker", mrkBlockEnd)
        #thisExp.addData("marker.time", blockendpulse_start_time)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blockEnd.maxDurationReached:
            routineTimer.addTime(-blockEnd.maxDuration)
        elif blockEnd.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "Confidencerating" ---
        # create an object to store info about Routine Confidencerating
        Confidencerating = data.Routine(
            name='Confidencerating',
            components=[text_confidencetask, slider_confidence],
        )
        Confidencerating.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        slider_confidence.reset()
        # Run 'Begin Routine' code from codeConfidenceRating
        # Makes mouse visible for this part of the experiment
        win.setMouseVisible(1)
        # store start times for Confidencerating
        Confidencerating.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Confidencerating.tStart = globalClock.getTime(format='float')
        Confidencerating.status = STARTED
        thisExp.addData('Confidencerating.started', Confidencerating.tStart)
        Confidencerating.maxDuration = None
        # keep track of which components have finished
        ConfidenceratingComponents = Confidencerating.components
        for thisComponent in Confidencerating.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Confidencerating" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        Confidencerating.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_confidencetask* updates
            
            # if text_confidencetask is starting this frame...
            if text_confidencetask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_confidencetask.frameNStart = frameN  # exact frame index
                text_confidencetask.tStart = t  # local t and not account for scr refresh
                text_confidencetask.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_confidencetask, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_confidencetask.started')
                # update status
                text_confidencetask.status = STARTED
                text_confidencetask.setAutoDraw(True)
            
            # if text_confidencetask is active this frame...
            if text_confidencetask.status == STARTED:
                # update params
                pass
            
            # *slider_confidence* updates
            
            # if slider_confidence is starting this frame...
            if slider_confidence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_confidence.frameNStart = frameN  # exact frame index
                slider_confidence.tStart = t  # local t and not account for scr refresh
                slider_confidence.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_confidence, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_confidence.started')
                # update status
                slider_confidence.status = STARTED
                slider_confidence.setAutoDraw(True)
            
            # if slider_confidence is active this frame...
            if slider_confidence.status == STARTED:
                # update params
                pass
            
            # Check slider_confidence for response to end Routine
            if slider_confidence.getRating() is not None and slider_confidence.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Confidencerating.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Confidencerating.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Confidencerating" ---
        for thisComponent in Confidencerating.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Confidencerating
        Confidencerating.tStop = globalClock.getTime(format='float')
        Confidencerating.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Confidencerating.stopped', Confidencerating.tStop)
        blocks.addData('slider_confidence.response', slider_confidence.getRating())
        blocks.addData('slider_confidence.rt', slider_confidence.getRT())
        # Run 'End Routine' code from codeConfidenceRating
        # Makes mouse invisible again
        win.setMouseVisible(0)
        # the Routine "Confidencerating" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed expInfo['blocks'] repeats of 'blocks'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if blocks.trialList in ([], [None], None):
        params = []
    else:
        params = blocks.trialList[0].keys()
    # save data for this loop
    blocks.saveAsExcel(filename + '.xlsx', sheetName='blocks',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # Run 'End Experiment' code from codeBlockStart
    ##### Uncomment the below for EEG Markers
    
    ## closing port again
    #port.close()
    # Run 'End Experiment' code from trial_code
    # This analyzes the participant's responses in the N-back task and logs the performance metrics.
    
    trialNone = 0; trialBack = 0; hit = 0; miss = 0; fa = 0
    for n, t in enumerate(thisExp.entries):
        if t['trial_type'] == 'none':
            trialNone += 1
            if not(t['trial_resp.corr']): fa += 1
        if t['trial_type'] == 'back':
            trialBack += 1
            if t['trial_resp.corr']: hit += 1
            else: miss += 1
    logging.log(level=logging.DATA, msg='hit={:.2f}%,miss={:.2f}%,fa={:.2f}%'.format(hit/trialBack*100,miss/trialBack*100,fa/trialNone*100))
    
    ##### Uncomment the below for EEG Markers
    
    ## closing port again
    #port.close()
    # Run 'End Experiment' code from codeBlockEnd
    ##### Uncomment the below for EEG Markers
    
    ## closing port again
    #port.close()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
