#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on February 04, 2025, at 22:26
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
prefs.hardware['audioLatencyMode'] = '3'
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
expName = 'GoNogo_EEG'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': 'subj-',
    'session': 'sess-01',
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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\pss1pd\\OneDrive - University of Surrey\\Documents\\Programs_Scripts\\PsychoPy\\GoNogo_EEG\\GoNogo_EEG.py',
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
            logging.getLevel('warning')
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
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1,-1,-1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
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
    if deviceManager.getDevice('key_resp_MainInstr') is None:
        # initialise key_resp_MainInstr
        key_resp_MainInstr = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_MainInstr',
        )
    if deviceManager.getDevice('key_resp_PractInstr') is None:
        # initialise key_resp_PractInstr
        key_resp_PractInstr = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_PractInstr',
        )
    if deviceManager.getDevice('key_resp_Prac') is None:
        # initialise key_resp_Prac
        key_resp_Prac = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_Prac',
        )
    if deviceManager.getDevice('key_resp_ExpInstr') is None:
        # initialise key_resp_ExpInstr
        key_resp_ExpInstr = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_ExpInstr',
        )
    if deviceManager.getDevice('key_resp_blockStart') is None:
        # initialise key_resp_blockStart
        key_resp_blockStart = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_blockStart',
        )
    if deviceManager.getDevice('key_resp_Main') is None:
        # initialise key_resp_Main
        key_resp_Main = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_Main',
        )
    if deviceManager.getDevice('key_resp_EndScreen') is None:
        # initialise key_resp_EndScreen
        key_resp_EndScreen = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_EndScreen',
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
    
    # --- Initialize components for Routine "mainInstr" ---
    text_MainInstr = visual.TextStim(win=win, name='text_MainInstr',
        text='Instructions\n\nIn this task you will be shown a series of circles, presented one at a time\n\nIf a Yellow Circle appears, Press the SPACEBAR as quickly as possible\n\nIf a Blue Circle appears, you should make NO RESPONSE\n\nTry to respond as quickly as possible, while still being accurate.\n\nWhen you are ready, \npress SPACEBAR to continue',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_MainInstr = keyboard.Keyboard(deviceName='key_resp_MainInstr')
    # Run 'Begin Experiment' code from code_Instr
    ## Experimental Setup
    # Setup Variables
    blockNum = 1
    
    
    # --- Initialize components for Routine "pracInstr" ---
    text_PractInstr = visual.TextStim(win=win, name='text_PractInstr',
        text='Practise Trials\n\nYou will now do some practise trials\n\nYou will get feedback after each trial \nabout your reaction time and accuracy\n\nPress SPACEBAR to begin',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_PractInstr = keyboard.Keyboard(deviceName='key_resp_PractInstr')
    
    # --- Initialize components for Routine "pracTrial" ---
    text_PreFixPrac = visual.TextStim(win=win, name='text_PreFixPrac',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_Prac = visual.ImageStim(
        win=win,
        name='image_Prac', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    text_PostFixPrac = visual.TextStim(win=win, name='text_PostFixPrac',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_Prac = keyboard.Keyboard(deviceName='key_resp_Prac')
    
    # --- Initialize components for Routine "pracFeedback" ---
    # Run 'Begin Experiment' code from code_Feedback
    #feedback variable just needs some value at start
    feedback=''
    text_Feedback = visual.TextStim(win=win, name='text_Feedback',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "expInstruct" ---
    text_ExpInstr = visual.TextStim(win=win, name='text_ExpInstr',
        text='Experimental Trials\n\nYou will now do the experimental trials\n\nThey are the same as the practise trials, \nbut you will not recieve feedback\n\nPress SPACEBAR to begin',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_ExpInstr = keyboard.Keyboard(deviceName='key_resp_ExpInstr')
    
    # --- Initialize components for Routine "blockStart" ---
    text_blockStart1 = visual.TextStim(win=win, name='text_blockStart1',
        text='Block:',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_blockStart2 = visual.TextStim(win=win, name='text_blockStart2',
        text=blockNum,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_blockStart3 = visual.TextStim(win=win, name='text_blockStart3',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_blockStart = keyboard.Keyboard(deviceName='key_resp_blockStart')
    
    # --- Initialize components for Routine "Trial" ---
    text_PreFixMain = visual.TextStim(win=win, name='text_PreFixMain',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_Main = visual.ImageStim(
        win=win,
        name='image_Main', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    text_PostFixMain = visual.TextStim(win=win, name='text_PostFixMain',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_Main = keyboard.Keyboard(deviceName='key_resp_Main')
    
    # --- Initialize components for Routine "blockEnd" ---
    text_blockEnd1 = visual.TextStim(win=win, name='text_blockEnd1',
        text='End of Block',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_blockEnd2 = visual.TextStim(win=win, name='text_blockEnd2',
        text=blockNum,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "endScreen" ---
    text_EndScreen = visual.TextStim(win=win, name='text_EndScreen',
        text='End of Experiment\n\nThank you for taking part!\n\nPress SPACE to end.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_EndScreen = keyboard.Keyboard(deviceName='key_resp_EndScreen')
    
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
    
    # --- Prepare to start Routine "mainInstr" ---
    # create an object to store info about Routine mainInstr
    mainInstr = data.Routine(
        name='mainInstr',
        components=[text_MainInstr, key_resp_MainInstr],
    )
    mainInstr.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_MainInstr
    key_resp_MainInstr.keys = []
    key_resp_MainInstr.rt = []
    _key_resp_MainInstr_allKeys = []
    # store start times for mainInstr
    mainInstr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    mainInstr.tStart = globalClock.getTime(format='float')
    mainInstr.status = STARTED
    thisExp.addData('mainInstr.started', mainInstr.tStart)
    mainInstr.maxDuration = None
    # keep track of which components have finished
    mainInstrComponents = mainInstr.components
    for thisComponent in mainInstr.components:
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
    
    # --- Run Routine "mainInstr" ---
    mainInstr.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_MainInstr* updates
        
        # if text_MainInstr is starting this frame...
        if text_MainInstr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_MainInstr.frameNStart = frameN  # exact frame index
            text_MainInstr.tStart = t  # local t and not account for scr refresh
            text_MainInstr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_MainInstr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_MainInstr.started')
            # update status
            text_MainInstr.status = STARTED
            text_MainInstr.setAutoDraw(True)
        
        # if text_MainInstr is active this frame...
        if text_MainInstr.status == STARTED:
            # update params
            pass
        
        # *key_resp_MainInstr* updates
        waitOnFlip = False
        
        # if key_resp_MainInstr is starting this frame...
        if key_resp_MainInstr.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_MainInstr.frameNStart = frameN  # exact frame index
            key_resp_MainInstr.tStart = t  # local t and not account for scr refresh
            key_resp_MainInstr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_MainInstr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_MainInstr.started')
            # update status
            key_resp_MainInstr.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_MainInstr.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_MainInstr.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_MainInstr.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_MainInstr.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_MainInstr_allKeys.extend(theseKeys)
            if len(_key_resp_MainInstr_allKeys):
                key_resp_MainInstr.keys = _key_resp_MainInstr_allKeys[-1].name  # just the last key pressed
                key_resp_MainInstr.rt = _key_resp_MainInstr_allKeys[-1].rt
                key_resp_MainInstr.duration = _key_resp_MainInstr_allKeys[-1].duration
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
            mainInstr.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in mainInstr.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "mainInstr" ---
    for thisComponent in mainInstr.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for mainInstr
    mainInstr.tStop = globalClock.getTime(format='float')
    mainInstr.tStopRefresh = tThisFlipGlobal
    thisExp.addData('mainInstr.stopped', mainInstr.tStop)
    # check responses
    if key_resp_MainInstr.keys in ['', [], None]:  # No response was made
        key_resp_MainInstr.keys = None
    thisExp.addData('key_resp_MainInstr.keys',key_resp_MainInstr.keys)
    if key_resp_MainInstr.keys != None:  # we had a response
        thisExp.addData('key_resp_MainInstr.rt', key_resp_MainInstr.rt)
        thisExp.addData('key_resp_MainInstr.duration', key_resp_MainInstr.duration)
    thisExp.nextEntry()
    # the Routine "mainInstr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pracInstr" ---
    # create an object to store info about Routine pracInstr
    pracInstr = data.Routine(
        name='pracInstr',
        components=[text_PractInstr, key_resp_PractInstr],
    )
    pracInstr.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_PractInstr
    key_resp_PractInstr.keys = []
    key_resp_PractInstr.rt = []
    _key_resp_PractInstr_allKeys = []
    # store start times for pracInstr
    pracInstr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pracInstr.tStart = globalClock.getTime(format='float')
    pracInstr.status = STARTED
    thisExp.addData('pracInstr.started', pracInstr.tStart)
    pracInstr.maxDuration = None
    # keep track of which components have finished
    pracInstrComponents = pracInstr.components
    for thisComponent in pracInstr.components:
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
    
    # --- Run Routine "pracInstr" ---
    pracInstr.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_PractInstr* updates
        
        # if text_PractInstr is starting this frame...
        if text_PractInstr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_PractInstr.frameNStart = frameN  # exact frame index
            text_PractInstr.tStart = t  # local t and not account for scr refresh
            text_PractInstr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_PractInstr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_PractInstr.started')
            # update status
            text_PractInstr.status = STARTED
            text_PractInstr.setAutoDraw(True)
        
        # if text_PractInstr is active this frame...
        if text_PractInstr.status == STARTED:
            # update params
            pass
        
        # *key_resp_PractInstr* updates
        waitOnFlip = False
        
        # if key_resp_PractInstr is starting this frame...
        if key_resp_PractInstr.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_PractInstr.frameNStart = frameN  # exact frame index
            key_resp_PractInstr.tStart = t  # local t and not account for scr refresh
            key_resp_PractInstr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_PractInstr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_PractInstr.started')
            # update status
            key_resp_PractInstr.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_PractInstr.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_PractInstr.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_PractInstr.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_PractInstr.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_PractInstr_allKeys.extend(theseKeys)
            if len(_key_resp_PractInstr_allKeys):
                key_resp_PractInstr.keys = _key_resp_PractInstr_allKeys[-1].name  # just the last key pressed
                key_resp_PractInstr.rt = _key_resp_PractInstr_allKeys[-1].rt
                key_resp_PractInstr.duration = _key_resp_PractInstr_allKeys[-1].duration
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
            pracInstr.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pracInstr.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pracInstr" ---
    for thisComponent in pracInstr.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pracInstr
    pracInstr.tStop = globalClock.getTime(format='float')
    pracInstr.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pracInstr.stopped', pracInstr.tStop)
    # check responses
    if key_resp_PractInstr.keys in ['', [], None]:  # No response was made
        key_resp_PractInstr.keys = None
    thisExp.addData('key_resp_PractInstr.keys',key_resp_PractInstr.keys)
    if key_resp_PractInstr.keys != None:  # we had a response
        thisExp.addData('key_resp_PractInstr.rt', key_resp_PractInstr.rt)
        thisExp.addData('key_resp_PractInstr.duration', key_resp_PractInstr.duration)
    thisExp.nextEntry()
    # the Routine "pracInstr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    pracTrials = data.TrialHandler2(
        name='pracTrials',
        nReps=2.0, 
        method='fullRandom', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Design/conditions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(pracTrials)  # add the loop to the experiment
    thisPracTrial = pracTrials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPracTrial.rgb)
    if thisPracTrial != None:
        for paramName in thisPracTrial:
            globals()[paramName] = thisPracTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPracTrial in pracTrials:
        currentLoop = pracTrials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPracTrial.rgb)
        if thisPracTrial != None:
            for paramName in thisPracTrial:
                globals()[paramName] = thisPracTrial[paramName]
        
        # --- Prepare to start Routine "pracTrial" ---
        # create an object to store info about Routine pracTrial
        pracTrial = data.Routine(
            name='pracTrial',
            components=[text_PreFixPrac, image_Prac, text_PostFixPrac, key_resp_Prac],
        )
        pracTrial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_Prac.setImage(GNG_Image)
        # create starting attributes for key_resp_Prac
        key_resp_Prac.keys = []
        key_resp_Prac.rt = []
        _key_resp_Prac_allKeys = []
        # store start times for pracTrial
        pracTrial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pracTrial.tStart = globalClock.getTime(format='float')
        pracTrial.status = STARTED
        thisExp.addData('pracTrial.started', pracTrial.tStart)
        pracTrial.maxDuration = None
        # keep track of which components have finished
        pracTrialComponents = pracTrial.components
        for thisComponent in pracTrial.components:
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
        
        # --- Run Routine "pracTrial" ---
        # if trial has changed, end Routine now
        if isinstance(pracTrials, data.TrialHandler2) and thisPracTrial.thisN != pracTrials.thisTrial.thisN:
            continueRoutine = False
        pracTrial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.2:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_PreFixPrac* updates
            
            # if text_PreFixPrac is starting this frame...
            if text_PreFixPrac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_PreFixPrac.frameNStart = frameN  # exact frame index
                text_PreFixPrac.tStart = t  # local t and not account for scr refresh
                text_PreFixPrac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_PreFixPrac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_PreFixPrac.started')
                # update status
                text_PreFixPrac.status = STARTED
                text_PreFixPrac.setAutoDraw(True)
            
            # if text_PreFixPrac is active this frame...
            if text_PreFixPrac.status == STARTED:
                # update params
                pass
            
            # if text_PreFixPrac is stopping this frame...
            if text_PreFixPrac.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_PreFixPrac.tStop = t  # not accounting for scr refresh
                    text_PreFixPrac.tStopRefresh = tThisFlipGlobal  # on global time
                    text_PreFixPrac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_PreFixPrac.stopped')
                    # update status
                    text_PreFixPrac.status = FINISHED
                    text_PreFixPrac.setAutoDraw(False)
            
            # *image_Prac* updates
            
            # if image_Prac is starting this frame...
            if image_Prac.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                image_Prac.frameNStart = frameN  # exact frame index
                image_Prac.tStart = t  # local t and not account for scr refresh
                image_Prac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_Prac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_Prac.started')
                # update status
                image_Prac.status = STARTED
                image_Prac.setAutoDraw(True)
            
            # if image_Prac is active this frame...
            if image_Prac.status == STARTED:
                # update params
                pass
            
            # if image_Prac is stopping this frame...
            if image_Prac.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    image_Prac.tStop = t  # not accounting for scr refresh
                    image_Prac.tStopRefresh = tThisFlipGlobal  # on global time
                    image_Prac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Prac.stopped')
                    # update status
                    image_Prac.status = FINISHED
                    image_Prac.setAutoDraw(False)
            
            # *text_PostFixPrac* updates
            
            # if text_PostFixPrac is starting this frame...
            if text_PostFixPrac.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_PostFixPrac.frameNStart = frameN  # exact frame index
                text_PostFixPrac.tStart = t  # local t and not account for scr refresh
                text_PostFixPrac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_PostFixPrac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_PostFixPrac.started')
                # update status
                text_PostFixPrac.status = STARTED
                text_PostFixPrac.setAutoDraw(True)
            
            # if text_PostFixPrac is active this frame...
            if text_PostFixPrac.status == STARTED:
                # update params
                pass
            
            # if text_PostFixPrac is stopping this frame...
            if text_PostFixPrac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_PostFixPrac.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    text_PostFixPrac.tStop = t  # not accounting for scr refresh
                    text_PostFixPrac.tStopRefresh = tThisFlipGlobal  # on global time
                    text_PostFixPrac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_PostFixPrac.stopped')
                    # update status
                    text_PostFixPrac.status = FINISHED
                    text_PostFixPrac.setAutoDraw(False)
            
            # *key_resp_Prac* updates
            waitOnFlip = False
            
            # if key_resp_Prac is starting this frame...
            if key_resp_Prac.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_resp_Prac.frameNStart = frameN  # exact frame index
                key_resp_Prac.tStart = t  # local t and not account for scr refresh
                key_resp_Prac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_Prac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_Prac.started')
                # update status
                key_resp_Prac.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_Prac.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_Prac.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_Prac is stopping this frame...
            if key_resp_Prac.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_Prac.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_Prac.tStop = t  # not accounting for scr refresh
                    key_resp_Prac.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_Prac.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_Prac.stopped')
                    # update status
                    key_resp_Prac.status = FINISHED
                    key_resp_Prac.status = FINISHED
            if key_resp_Prac.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_Prac.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_Prac_allKeys.extend(theseKeys)
                if len(_key_resp_Prac_allKeys):
                    key_resp_Prac.keys = _key_resp_Prac_allKeys[-1].name  # just the last key pressed
                    key_resp_Prac.rt = _key_resp_Prac_allKeys[-1].rt
                    key_resp_Prac.duration = _key_resp_Prac_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_Prac.keys == str(GNG_CorrectResp)) or (key_resp_Prac.keys == GNG_CorrectResp):
                        key_resp_Prac.corr = 1
                    else:
                        key_resp_Prac.corr = 0
            
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
                pracTrial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pracTrial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pracTrial" ---
        for thisComponent in pracTrial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pracTrial
        pracTrial.tStop = globalClock.getTime(format='float')
        pracTrial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pracTrial.stopped', pracTrial.tStop)
        # check responses
        if key_resp_Prac.keys in ['', [], None]:  # No response was made
            key_resp_Prac.keys = None
            # was no response the correct answer?!
            if str(GNG_CorrectResp).lower() == 'none':
               key_resp_Prac.corr = 1;  # correct non-response
            else:
               key_resp_Prac.corr = 0;  # failed to respond (incorrectly)
        # store data for pracTrials (TrialHandler)
        pracTrials.addData('key_resp_Prac.keys',key_resp_Prac.keys)
        pracTrials.addData('key_resp_Prac.corr', key_resp_Prac.corr)
        if key_resp_Prac.keys != None:  # we had a response
            pracTrials.addData('key_resp_Prac.rt', key_resp_Prac.rt)
            pracTrials.addData('key_resp_Prac.duration', key_resp_Prac.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if pracTrial.maxDurationReached:
            routineTimer.addTime(-pracTrial.maxDuration)
        elif pracTrial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.200000)
        
        # --- Prepare to start Routine "pracFeedback" ---
        # create an object to store info about Routine pracFeedback
        pracFeedback = data.Routine(
            name='pracFeedback',
            components=[text_Feedback],
        )
        pracFeedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_Feedback
        if GNG_Condition == "NoGo" :
            if not key_resp_Prac.keys :
                feedback="Correct!"
            else: 
                feedback="Wrong. No response required"
        
        if GNG_Condition == "Go" :
            if not key_resp_Prac.keys :
                feedback="Wrong. Failed to respond"
            elif key_resp_Prac.corr:
                feedback="Correct! RT=%.3f" %(key_resp_Prac.rt)
            else: 
                feedback="Wrong. Incorrect Response"
        text_Feedback.setText(feedback)
        # store start times for pracFeedback
        pracFeedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pracFeedback.tStart = globalClock.getTime(format='float')
        pracFeedback.status = STARTED
        thisExp.addData('pracFeedback.started', pracFeedback.tStart)
        pracFeedback.maxDuration = None
        # keep track of which components have finished
        pracFeedbackComponents = pracFeedback.components
        for thisComponent in pracFeedback.components:
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
        
        # --- Run Routine "pracFeedback" ---
        # if trial has changed, end Routine now
        if isinstance(pracTrials, data.TrialHandler2) and thisPracTrial.thisN != pracTrials.thisTrial.thisN:
            continueRoutine = False
        pracFeedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_Feedback* updates
            
            # if text_Feedback is starting this frame...
            if text_Feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_Feedback.frameNStart = frameN  # exact frame index
                text_Feedback.tStart = t  # local t and not account for scr refresh
                text_Feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_Feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_Feedback.started')
                # update status
                text_Feedback.status = STARTED
                text_Feedback.setAutoDraw(True)
            
            # if text_Feedback is active this frame...
            if text_Feedback.status == STARTED:
                # update params
                pass
            
            # if text_Feedback is stopping this frame...
            if text_Feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_Feedback.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_Feedback.tStop = t  # not accounting for scr refresh
                    text_Feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    text_Feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_Feedback.stopped')
                    # update status
                    text_Feedback.status = FINISHED
                    text_Feedback.setAutoDraw(False)
            
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
                pracFeedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pracFeedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pracFeedback" ---
        for thisComponent in pracFeedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pracFeedback
        pracFeedback.tStop = globalClock.getTime(format='float')
        pracFeedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pracFeedback.stopped', pracFeedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if pracFeedback.maxDurationReached:
            routineTimer.addTime(-pracFeedback.maxDuration)
        elif pracFeedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'pracTrials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "expInstruct" ---
    # create an object to store info about Routine expInstruct
    expInstruct = data.Routine(
        name='expInstruct',
        components=[text_ExpInstr, key_resp_ExpInstr],
    )
    expInstruct.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_ExpInstr
    key_resp_ExpInstr.keys = []
    key_resp_ExpInstr.rt = []
    _key_resp_ExpInstr_allKeys = []
    # store start times for expInstruct
    expInstruct.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    expInstruct.tStart = globalClock.getTime(format='float')
    expInstruct.status = STARTED
    thisExp.addData('expInstruct.started', expInstruct.tStart)
    expInstruct.maxDuration = None
    # keep track of which components have finished
    expInstructComponents = expInstruct.components
    for thisComponent in expInstruct.components:
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
    
    # --- Run Routine "expInstruct" ---
    expInstruct.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_ExpInstr* updates
        
        # if text_ExpInstr is starting this frame...
        if text_ExpInstr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_ExpInstr.frameNStart = frameN  # exact frame index
            text_ExpInstr.tStart = t  # local t and not account for scr refresh
            text_ExpInstr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ExpInstr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ExpInstr.started')
            # update status
            text_ExpInstr.status = STARTED
            text_ExpInstr.setAutoDraw(True)
        
        # if text_ExpInstr is active this frame...
        if text_ExpInstr.status == STARTED:
            # update params
            pass
        
        # *key_resp_ExpInstr* updates
        waitOnFlip = False
        
        # if key_resp_ExpInstr is starting this frame...
        if key_resp_ExpInstr.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_ExpInstr.frameNStart = frameN  # exact frame index
            key_resp_ExpInstr.tStart = t  # local t and not account for scr refresh
            key_resp_ExpInstr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_ExpInstr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_ExpInstr.started')
            # update status
            key_resp_ExpInstr.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_ExpInstr.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_ExpInstr.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_ExpInstr.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_ExpInstr.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_ExpInstr_allKeys.extend(theseKeys)
            if len(_key_resp_ExpInstr_allKeys):
                key_resp_ExpInstr.keys = _key_resp_ExpInstr_allKeys[-1].name  # just the last key pressed
                key_resp_ExpInstr.rt = _key_resp_ExpInstr_allKeys[-1].rt
                key_resp_ExpInstr.duration = _key_resp_ExpInstr_allKeys[-1].duration
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
            expInstruct.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in expInstruct.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "expInstruct" ---
    for thisComponent in expInstruct.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for expInstruct
    expInstruct.tStop = globalClock.getTime(format='float')
    expInstruct.tStopRefresh = tThisFlipGlobal
    thisExp.addData('expInstruct.stopped', expInstruct.tStop)
    # check responses
    if key_resp_ExpInstr.keys in ['', [], None]:  # No response was made
        key_resp_ExpInstr.keys = None
    thisExp.addData('key_resp_ExpInstr.keys',key_resp_ExpInstr.keys)
    if key_resp_ExpInstr.keys != None:  # we had a response
        thisExp.addData('key_resp_ExpInstr.rt', key_resp_ExpInstr.rt)
        thisExp.addData('key_resp_ExpInstr.duration', key_resp_ExpInstr.duration)
    thisExp.nextEntry()
    # the Routine "expInstruct" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=2.0, 
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
            components=[text_blockStart1, text_blockStart2, text_blockStart3, key_resp_blockStart],
        )
        blockStart.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_blockStart
        key_resp_blockStart.keys = []
        key_resp_blockStart.rt = []
        _key_resp_blockStart_allKeys = []
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
            
            # *text_blockStart1* updates
            
            # if text_blockStart1 is starting this frame...
            if text_blockStart1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_blockStart1.frameNStart = frameN  # exact frame index
                text_blockStart1.tStart = t  # local t and not account for scr refresh
                text_blockStart1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_blockStart1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_blockStart1.started')
                # update status
                text_blockStart1.status = STARTED
                text_blockStart1.setAutoDraw(True)
            
            # if text_blockStart1 is active this frame...
            if text_blockStart1.status == STARTED:
                # update params
                pass
            
            # *text_blockStart2* updates
            
            # if text_blockStart2 is starting this frame...
            if text_blockStart2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_blockStart2.frameNStart = frameN  # exact frame index
                text_blockStart2.tStart = t  # local t and not account for scr refresh
                text_blockStart2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_blockStart2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_blockStart2.started')
                # update status
                text_blockStart2.status = STARTED
                text_blockStart2.setAutoDraw(True)
            
            # if text_blockStart2 is active this frame...
            if text_blockStart2.status == STARTED:
                # update params
                pass
            
            # *text_blockStart3* updates
            
            # if text_blockStart3 is starting this frame...
            if text_blockStart3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_blockStart3.frameNStart = frameN  # exact frame index
                text_blockStart3.tStart = t  # local t and not account for scr refresh
                text_blockStart3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_blockStart3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_blockStart3.started')
                # update status
                text_blockStart3.status = STARTED
                text_blockStart3.setAutoDraw(True)
            
            # if text_blockStart3 is active this frame...
            if text_blockStart3.status == STARTED:
                # update params
                pass
            
            # *key_resp_blockStart* updates
            waitOnFlip = False
            
            # if key_resp_blockStart is starting this frame...
            if key_resp_blockStart.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_blockStart.frameNStart = frameN  # exact frame index
                key_resp_blockStart.tStart = t  # local t and not account for scr refresh
                key_resp_blockStart.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_blockStart, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_blockStart.started')
                # update status
                key_resp_blockStart.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_blockStart.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_blockStart.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_blockStart.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_blockStart.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_blockStart_allKeys.extend(theseKeys)
                if len(_key_resp_blockStart_allKeys):
                    key_resp_blockStart.keys = _key_resp_blockStart_allKeys[-1].name  # just the last key pressed
                    key_resp_blockStart.rt = _key_resp_blockStart_allKeys[-1].rt
                    key_resp_blockStart.duration = _key_resp_blockStart_allKeys[-1].duration
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
        # check responses
        if key_resp_blockStart.keys in ['', [], None]:  # No response was made
            key_resp_blockStart.keys = None
        blocks.addData('key_resp_blockStart.keys',key_resp_blockStart.keys)
        if key_resp_blockStart.keys != None:  # we had a response
            blocks.addData('key_resp_blockStart.rt', key_resp_blockStart.rt)
            blocks.addData('key_resp_blockStart.duration', key_resp_blockStart.duration)
        # the Routine "blockStart" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        mainTrials = data.TrialHandler2(
            name='mainTrials',
            nReps=16.0, 
            method='fullRandom', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('Design/conditions.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(mainTrials)  # add the loop to the experiment
        thisMainTrial = mainTrials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMainTrial.rgb)
        if thisMainTrial != None:
            for paramName in thisMainTrial:
                globals()[paramName] = thisMainTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisMainTrial in mainTrials:
            currentLoop = mainTrials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisMainTrial.rgb)
            if thisMainTrial != None:
                for paramName in thisMainTrial:
                    globals()[paramName] = thisMainTrial[paramName]
            
            # --- Prepare to start Routine "Trial" ---
            # create an object to store info about Routine Trial
            Trial = data.Routine(
                name='Trial',
                components=[text_PreFixMain, image_Main, text_PostFixMain, key_resp_Main],
            )
            Trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_Main.setImage(GNG_Image)
            # create starting attributes for key_resp_Main
            key_resp_Main.keys = []
            key_resp_Main.rt = []
            _key_resp_Main_allKeys = []
            # store start times for Trial
            Trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Trial.tStart = globalClock.getTime(format='float')
            Trial.status = STARTED
            thisExp.addData('Trial.started', Trial.tStart)
            Trial.maxDuration = None
            # keep track of which components have finished
            TrialComponents = Trial.components
            for thisComponent in Trial.components:
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
            
            # --- Run Routine "Trial" ---
            # if trial has changed, end Routine now
            if isinstance(mainTrials, data.TrialHandler2) and thisMainTrial.thisN != mainTrials.thisTrial.thisN:
                continueRoutine = False
            Trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.2:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_PreFixMain* updates
                
                # if text_PreFixMain is starting this frame...
                if text_PreFixMain.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_PreFixMain.frameNStart = frameN  # exact frame index
                    text_PreFixMain.tStart = t  # local t and not account for scr refresh
                    text_PreFixMain.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_PreFixMain, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_PreFixMain.started')
                    # update status
                    text_PreFixMain.status = STARTED
                    text_PreFixMain.setAutoDraw(True)
                
                # if text_PreFixMain is active this frame...
                if text_PreFixMain.status == STARTED:
                    # update params
                    pass
                
                # if text_PreFixMain is stopping this frame...
                if text_PreFixMain.status == STARTED:
                    # is it time to stop? (based on local clock)
                    if tThisFlip > 0.2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_PreFixMain.tStop = t  # not accounting for scr refresh
                        text_PreFixMain.tStopRefresh = tThisFlipGlobal  # on global time
                        text_PreFixMain.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_PreFixMain.stopped')
                        # update status
                        text_PreFixMain.status = FINISHED
                        text_PreFixMain.setAutoDraw(False)
                
                # *image_Main* updates
                
                # if image_Main is starting this frame...
                if image_Main.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                    # keep track of start time/frame for later
                    image_Main.frameNStart = frameN  # exact frame index
                    image_Main.tStart = t  # local t and not account for scr refresh
                    image_Main.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_Main, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_Main.started')
                    # update status
                    image_Main.status = STARTED
                    image_Main.setAutoDraw(True)
                
                # if image_Main is active this frame...
                if image_Main.status == STARTED:
                    # update params
                    pass
                
                # if image_Main is stopping this frame...
                if image_Main.status == STARTED:
                    # is it time to stop? (based on local clock)
                    if tThisFlip > 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        image_Main.tStop = t  # not accounting for scr refresh
                        image_Main.tStopRefresh = tThisFlipGlobal  # on global time
                        image_Main.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_Main.stopped')
                        # update status
                        image_Main.status = FINISHED
                        image_Main.setAutoDraw(False)
                
                # *text_PostFixMain* updates
                
                # if text_PostFixMain is starting this frame...
                if text_PostFixMain.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    text_PostFixMain.frameNStart = frameN  # exact frame index
                    text_PostFixMain.tStart = t  # local t and not account for scr refresh
                    text_PostFixMain.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_PostFixMain, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_PostFixMain.started')
                    # update status
                    text_PostFixMain.status = STARTED
                    text_PostFixMain.setAutoDraw(True)
                
                # if text_PostFixMain is active this frame...
                if text_PostFixMain.status == STARTED:
                    # update params
                    pass
                
                # if text_PostFixMain is stopping this frame...
                if text_PostFixMain.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_PostFixMain.tStartRefresh + 0.7-frameTolerance:
                        # keep track of stop time/frame for later
                        text_PostFixMain.tStop = t  # not accounting for scr refresh
                        text_PostFixMain.tStopRefresh = tThisFlipGlobal  # on global time
                        text_PostFixMain.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_PostFixMain.stopped')
                        # update status
                        text_PostFixMain.status = FINISHED
                        text_PostFixMain.setAutoDraw(False)
                
                # *key_resp_Main* updates
                waitOnFlip = False
                
                # if key_resp_Main is starting this frame...
                if key_resp_Main.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_Main.frameNStart = frameN  # exact frame index
                    key_resp_Main.tStart = t  # local t and not account for scr refresh
                    key_resp_Main.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_Main, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_Main.started')
                    # update status
                    key_resp_Main.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_Main.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_Main.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_Main is stopping this frame...
                if key_resp_Main.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_Main.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_Main.tStop = t  # not accounting for scr refresh
                        key_resp_Main.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_Main.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_Main.stopped')
                        # update status
                        key_resp_Main.status = FINISHED
                        key_resp_Main.status = FINISHED
                if key_resp_Main.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_Main.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_Main_allKeys.extend(theseKeys)
                    if len(_key_resp_Main_allKeys):
                        key_resp_Main.keys = _key_resp_Main_allKeys[-1].name  # just the last key pressed
                        key_resp_Main.rt = _key_resp_Main_allKeys[-1].rt
                        key_resp_Main.duration = _key_resp_Main_allKeys[-1].duration
                        # was this correct?
                        if (key_resp_Main.keys == str(GNG_CorrectResp)) or (key_resp_Main.keys == GNG_CorrectResp):
                            key_resp_Main.corr = 1
                        else:
                            key_resp_Main.corr = 0
                
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
                    Trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Trial" ---
            for thisComponent in Trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Trial
            Trial.tStop = globalClock.getTime(format='float')
            Trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Trial.stopped', Trial.tStop)
            # check responses
            if key_resp_Main.keys in ['', [], None]:  # No response was made
                key_resp_Main.keys = None
                # was no response the correct answer?!
                if str(GNG_CorrectResp).lower() == 'none':
                   key_resp_Main.corr = 1;  # correct non-response
                else:
                   key_resp_Main.corr = 0;  # failed to respond (incorrectly)
            # store data for mainTrials (TrialHandler)
            mainTrials.addData('key_resp_Main.keys',key_resp_Main.keys)
            mainTrials.addData('key_resp_Main.corr', key_resp_Main.corr)
            if key_resp_Main.keys != None:  # we had a response
                mainTrials.addData('key_resp_Main.rt', key_resp_Main.rt)
                mainTrials.addData('key_resp_Main.duration', key_resp_Main.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Trial.maxDurationReached:
                routineTimer.addTime(-Trial.maxDuration)
            elif Trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.200000)
            thisExp.nextEntry()
            
        # completed 16.0 repeats of 'mainTrials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "blockEnd" ---
        # create an object to store info about Routine blockEnd
        blockEnd = data.Routine(
            name='blockEnd',
            components=[text_blockEnd1, text_blockEnd2],
        )
        blockEnd.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
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
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_blockEnd1* updates
            
            # if text_blockEnd1 is starting this frame...
            if text_blockEnd1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_blockEnd1.frameNStart = frameN  # exact frame index
                text_blockEnd1.tStart = t  # local t and not account for scr refresh
                text_blockEnd1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_blockEnd1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_blockEnd1.started')
                # update status
                text_blockEnd1.status = STARTED
                text_blockEnd1.setAutoDraw(True)
            
            # if text_blockEnd1 is active this frame...
            if text_blockEnd1.status == STARTED:
                # update params
                pass
            
            # if text_blockEnd1 is stopping this frame...
            if text_blockEnd1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_blockEnd1.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_blockEnd1.tStop = t  # not accounting for scr refresh
                    text_blockEnd1.tStopRefresh = tThisFlipGlobal  # on global time
                    text_blockEnd1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blockEnd1.stopped')
                    # update status
                    text_blockEnd1.status = FINISHED
                    text_blockEnd1.setAutoDraw(False)
            
            # *text_blockEnd2* updates
            
            # if text_blockEnd2 is starting this frame...
            if text_blockEnd2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_blockEnd2.frameNStart = frameN  # exact frame index
                text_blockEnd2.tStart = t  # local t and not account for scr refresh
                text_blockEnd2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_blockEnd2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_blockEnd2.started')
                # update status
                text_blockEnd2.status = STARTED
                text_blockEnd2.setAutoDraw(True)
            
            # if text_blockEnd2 is active this frame...
            if text_blockEnd2.status == STARTED:
                # update params
                pass
            
            # if text_blockEnd2 is stopping this frame...
            if text_blockEnd2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_blockEnd2.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_blockEnd2.tStop = t  # not accounting for scr refresh
                    text_blockEnd2.tStopRefresh = tThisFlipGlobal  # on global time
                    text_blockEnd2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blockEnd2.stopped')
                    # update status
                    text_blockEnd2.status = FINISHED
                    text_blockEnd2.setAutoDraw(False)
            
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
        # Run 'End Routine' code from code_blockEnd
        ## iterate block number for block start text
        blockNum = blockNum +1
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blockEnd.maxDurationReached:
            routineTimer.addTime(-blockEnd.maxDuration)
        elif blockEnd.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'blocks'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "endScreen" ---
    # create an object to store info about Routine endScreen
    endScreen = data.Routine(
        name='endScreen',
        components=[text_EndScreen, key_resp_EndScreen],
    )
    endScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_EndScreen
    key_resp_EndScreen.keys = []
    key_resp_EndScreen.rt = []
    _key_resp_EndScreen_allKeys = []
    # store start times for endScreen
    endScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    endScreen.tStart = globalClock.getTime(format='float')
    endScreen.status = STARTED
    thisExp.addData('endScreen.started', endScreen.tStart)
    endScreen.maxDuration = None
    # keep track of which components have finished
    endScreenComponents = endScreen.components
    for thisComponent in endScreen.components:
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
    
    # --- Run Routine "endScreen" ---
    endScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_EndScreen* updates
        
        # if text_EndScreen is starting this frame...
        if text_EndScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_EndScreen.frameNStart = frameN  # exact frame index
            text_EndScreen.tStart = t  # local t and not account for scr refresh
            text_EndScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_EndScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_EndScreen.started')
            # update status
            text_EndScreen.status = STARTED
            text_EndScreen.setAutoDraw(True)
        
        # if text_EndScreen is active this frame...
        if text_EndScreen.status == STARTED:
            # update params
            pass
        
        # *key_resp_EndScreen* updates
        waitOnFlip = False
        
        # if key_resp_EndScreen is starting this frame...
        if key_resp_EndScreen.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            # keep track of start time/frame for later
            key_resp_EndScreen.frameNStart = frameN  # exact frame index
            key_resp_EndScreen.tStart = t  # local t and not account for scr refresh
            key_resp_EndScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_EndScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_EndScreen.started')
            # update status
            key_resp_EndScreen.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_EndScreen.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_EndScreen.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_EndScreen.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_EndScreen.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_EndScreen_allKeys.extend(theseKeys)
            if len(_key_resp_EndScreen_allKeys):
                key_resp_EndScreen.keys = _key_resp_EndScreen_allKeys[-1].name  # just the last key pressed
                key_resp_EndScreen.rt = _key_resp_EndScreen_allKeys[-1].rt
                key_resp_EndScreen.duration = _key_resp_EndScreen_allKeys[-1].duration
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
            endScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "endScreen" ---
    for thisComponent in endScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for endScreen
    endScreen.tStop = globalClock.getTime(format='float')
    endScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('endScreen.stopped', endScreen.tStop)
    # check responses
    if key_resp_EndScreen.keys in ['', [], None]:  # No response was made
        key_resp_EndScreen.keys = None
    thisExp.addData('key_resp_EndScreen.keys',key_resp_EndScreen.keys)
    if key_resp_EndScreen.keys != None:  # we had a response
        thisExp.addData('key_resp_EndScreen.rt', key_resp_EndScreen.rt)
        thisExp.addData('key_resp_EndScreen.duration', key_resp_EndScreen.duration)
    thisExp.nextEntry()
    # the Routine "endScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
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
