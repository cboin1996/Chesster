 Chesster

### About:
Chesster is a chess AI project.

### Install:
git clone https://github.com/cboin1996/Chesster.git

cd into Chesster
configure your virtual environment
```bash
	python -m venv venv
	venv/Script/activate (windows)
	source ./venv/bin/activate (linux/macos)
```  

Now install the requirements from requirements.txt
```bash
pip install requirements.txt
```
Running the reinforcement learning program:  
```bash
python reinforce_run.py --cmd [options: lic, sp, sl, eval, tr]
```

### Command line args
- [1]lic: lichess bot
- [2]sp: self play module 
- [3]sl: supervised learning module
- [4]tr: training module
- [5]eval: model evaluator module

If you want to train a model from PGN data, copy PGN file into play_data.    
- run [3] to generate play files.
- run [4] to train off the play files
- run [5] to evaluate which model is best 

To run in reinforcement learning mode.. run the same as above but with [2] instead of 3.  

- [1] was created as an option to play against your engine using lichess. In order to use this module, create a .json file named secret with at minimum the following string {"token" : "your-api-key"}.  
 

The deep_learning library is an implementation of https://github.com/Zeta36/chess-alpha-zero

