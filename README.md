# Keystroke Authenticator

A full-stack project for authenticating users based on their typing patterns (keystroke dynamics) using machine learning. The project consists of a Python back-end for data collection, processing, and model training, and a modern JavaScript front-end for user interaction.

## Features
- Collects and processes keystroke data
- Trains machine learning models to recognize users by typing style
- REST API for authentication and data submission
- Front-end UI for data entry and authentication

## Project Structure
```
keystroke-authenticator/
│
├── back-end/                # Python backend
│   ├── collection.py        # Data collection scripts
│   ├── generate_data.py     # Data generation utilities
│   ├── requirements.txt     # Python dependencies
│   ├── ml_pipeline/         # ML training and evaluation
│   ├── processed_data/      # Processed datasets
│   ├── saved_models/        # Trained models
│   └── keystroke_data/      # Raw keystroke CSVs
│
├── front-end/               # Frontend (React + Tailwind CSS)
│   ├── package.json         # JS dependencies
│   ├── src/                 # Source code
│   ├── public/              # Static assets
│   └── ...
│
└── README.md                # Project documentation
```

## Getting Started

### Back-End (Python)
1. Install Python 3.11+
2. Install dependencies:
   ```sh
   pip install -r back-end/requirements.txt
   ```
3. Run backend using python collection.py

### Front-End (React)
1. Install Node.js (v18+ recommended)
2. Install dependencies:
   ```sh
   cd front-end
   npm install
   ```
3. Start the development server:
   ```sh
   npm run dev
   ```
4. Make sure to run backend and frontend at the same time in different shells to get full functionality

## CUDA & GPU Support
- Ensure you have an NVIDIA GPU with up-to-date drivers.
- (Optional) Install the CUDA Toolkit matching your PyTorch version (see [PyTorch Get Started](https://pytorch.org/get-started/locally/)).

## Usage
- Use the front-end to collect keystroke data and authenticate users.
- Use the back-end scripts to process data and train models.
- Integrate the front-end and back-end as needed for your deployment.
