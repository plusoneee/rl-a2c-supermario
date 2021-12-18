# RL: A2C Super-mario-bros

## Getting Started
* The project use `PyTorch` to implement, make sure `torch` is installed.

### Requirements
* Install `Python3` & `pip` first. 
* Then, run:
```
pip install -r requirements.txt
```

## Source Code Structure

* `memory` implements mamory object.
* `networks` defines the CNN architecture and its parameters.
* `mario` implements the anythings the agent could do here (like get action, compute loss).
* `main` glues everything together to implement the A2C algorithm.
* `wrappers` for custom the enviroment.

## Run Training
```
python main.py
```
