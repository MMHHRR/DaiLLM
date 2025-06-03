# LLM-GAN Human Mobility Generator

This project implements a GAN-like architecture using Large Language Models (LLMs) to generate human mobility trajectories. The system consists of three main components:

1. Profiler: Analyzes historical and current trajectory data to create user profiles
2. Generator: Creates new trajectories based on the profiles
3. Discriminator: Evaluates the generated trajectories against real data

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Prepare your input data in CSV format with the following columns:
   - venueCategory
   - latitude
   - longitude
   - timezoneOffset
   - utcTimestamp

2. Run the main script:
```bash
python llm_gan_mobility.py
```

The script will:
- Analyze the input data
- Generate new trajectories
- Evaluate the results
- Save the generated trajectories to `generated_trajectory.csv`

## Components

### Profiler
- Analyzes long-term user profiles
- Identifies short-term mobility patterns
- Uses GPT-4 for pattern recognition

### Generator
- Creates new trajectories based on profiles
- Maintains temporal and spatial consistency
- Generates realistic venue sequences

### Discriminator
- Evaluates generated trajectories
- Checks temporal patterns
- Verifies venue type frequency
- Validates geographical distribution
- Ensures logical venue transitions
- Confirms appropriate stay durations

## Output

The system generates a CSV file containing:
- Generated venue categories
- Timestamps
- Coordinates
- Timezone information 