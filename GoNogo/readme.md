# Go Nogo

This task is a Go-Nogo Task in which subjects have to respond as quickly as possible to target stimuli and withhold responses to distractor (non-target) stimuli. 

It is typically used to measure inhibitory control, specifically response inhibition.

A basic version is coded here:

		Target Stimulus: blue circle
		Non-Target Stimulus: yellow circle
		Stimulus Presentation Time (SPT): 300 milliseconds
		Inter-Stimulus Intervals (ISI): 900 milliseconds
		Target Probability:20%
		80 Trials per Block: 16 Target, 64 Non-Target
		2 Blocks

Participants first take part in a practise session of 10 trials (2 Target, 8 Non-Target).

## More Design information

The Go Nogo is coded from an excel file with 5 trials (4 Go, 1 Nogo), and these are presented in "Full Random" order, so if you ask for 2 repetitions, PsychoPy treats this as 8 Go and 2 NoGo and Randomises this. 
 
Within the startup menu you can change:

- the number of repetitions for Practise Trials (currently 2 = 10 trials). If you put as 0, then this will skip practise. 

- the number of repetitions for main trials (currently 16, which makes 80 trials per block).

- the number of blocks (currently 2, each block approx 1 minute 36 long [for 80 trials]). 

## *Based On:* 

Mehrnaz Rezvanfard, Mehrshad Golesorkhi, Ensieh Ghassemian, Hooman Safaei, Aiden Nasiri Eghbali, Hanieh Alizadeh, Hamed Ekhtiari. (2016). Evaluation of Inhibition Response Behaviour Using the Go/No-Go Paradigm in Normal Individuals: Effects of Variations in the Task Design. Acta Neuropsychologia. Vol. 14, No. 4, 2016, 357-366

(their Go/No-Go tasks were developed using E-Prime V.2 software)