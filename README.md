# mmGomoku

## Introduction

This repository presents an AI designed specifically for the game of Gomoku. The design philosophy and implementation of the AI is twofold:

1. **Core Engine**: Written in C++, it encompasses a search engine tailored for the basic Gomoku board patterns.
2. **High-Level AI**: This is constructed using Python and involves:
   - **Monte Carlo Tree Search (MCTS)** algorithm, borrowed and adapted from the Alpha Go strategy.
   - A suitably adjusted neural network which aids in achieving higher intelligence in the game's decision-making.

Additionally, this repository is a refactor of the earlier project found here: [alpha_gomoku](https://github.com/YouHuang67/alpha_gomoku).

## TODO List

- [ ] Documentation Writing
- [ ] Migration of Old Code
- [ ] Training Code Optimization


## Installation

Follow these steps to install the necessary environment for this project.

### 1. Python Environment

Ensure you're using Python 3.11.0. You can set up a virtual environment with this specific version using the following commands (assuming `python3.11` is installed):

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 2. Install PyTorch

Depending on your setup (CUDA version or CPU-only), use one of the following commands to install PyTorch 2.0.1 along with `torchvision` and `torchaudio`.

#### For CUDA 11.7:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

#### For CUDA 11.8:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

#### For CPU-only:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Required Python Packages

Next, install all other required Python packages by using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Install MMsegmentation and Dependencies

To install the necessary `mmsegmentation` and its dependencies:

```bash
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.1
```

Then install `mmsegmentation`:

```bash
cd mmsegmentation
pip install -e .
cd ..
```

### 5. Install Gomoku C++ Module (for Efficient Pattern Searching)

To install the Gomoku C++ module, follow these steps to build and place the compiled library outside of the `projects/cppboard/bitboard` directory:

```bash
cd projects/cppboard/bitboard
python setup.py build_ext --build-lib=../
```

After completing these steps, the C++ module will be installed, and you'll be able to use it for efficient searching in your project.

That's it! You should now have everything set up.

## Usage
_TODO: Provide detailed instructions on how to use the Gomoku AI, including any example commands or scripts._

## Contributors
_TODO: List main contributors or provide instructions on how to contribute to the project._

## License
_TODO: Add licensing details or link to a LICENSE file._

For further details and queries, please feel free to open an issue or contact the maintainer.
