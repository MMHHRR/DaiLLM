import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import json

# Load environment variables
load_dotenv()
client = openai.OpenAI(
    api_key="sk-YSKJPd8GuwHqxMaFrbMU50sjJ4AVJNVftjuCUCZvcvwaWl3k",
    base_url="https://api.zetatechs.com/v1/"
)

class Profiler:
    def __init__(self):
        self.long_term_profile = None
        self.short_term_pattern = None
    
    def analyze_long_term_profile(self, historical_trajectory: pd.DataFrame) -> str:
        prompt = f"""
        Based on the following historical trajectory data, analyze the user's profile:
        
        Historical Trajectory:
        {historical_trajectory.to_string()}
        
        Please analyze and provide limited 100 words:
        1. Gender (male or female)
        2. Age group estimation
        3. Race inference
        4. Income level estimation
        5. Likely occupation
        6. Lifestyle characteristics
        7. Regular activity patterns
        8. Preferred locations and venue types
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        self.long_term_profile = response.choices[0].message.content
        return self.long_term_profile
    
    def analyze_short_term_pattern(self, historical_trajectory: pd.DataFrame) -> str:
        prompt = f"""
        Analyze the following current trajectory data to identify mobility patterns:
        
        Historical Trajectory:
        {historical_trajectory.to_string()}
        
        Please identify, limited 200 words:
        1. Peak activity periods
        2. Key destinations
        3. Daily routes
        4. Temporal patterns
        5. Categorical preferences
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        self.short_term_pattern = response.choices[0].message.content
        return self.short_term_pattern

class Generator:
    def __init__(self):
        self.generated_trajectory = None
    
    def generate_trajectory(self, profile: str, pattern: str, historical_data: pd.DataFrame, feedback_prompt: str = None) -> pd.DataFrame:
        prompt = f"""
        Generate a one-day from morning to night trajectory based on:
        
        User Profile:
        {profile}
        
        Mobility Pattern:
        {pattern}
        
        Historical Data:
        {historical_data.to_string()}
        
        Generate a realistic trajectory that includes:
        1. Venue categories
        2. Timestamps
        3. Latitude and longitude coordinates
        4. Timezone information
        
        Return the trajectory in the following JSON format:
        {{
            "trajectory": [
                {{
                    "timestamp": "YYYY-MM-DD HH:MM:SS",
                    "venue_category": "string",
                    "latitude": float,
                    "longitude": float,
                    "timezone_offset": integer
                }},
                ...
            ]
        }}
        
        Note: Only output the JSON, no other text. Do not include markdown code block markers.
        """
        
        if feedback_prompt:
            prompt += f"\n\nPrevious feedback to consider:\n{feedback_prompt}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            trajectory_data = json.loads(response.choices[0].message.content)['trajectory']
            dtype_map = {
                'timestamp': 'datetime64[ns]',
                'venue_category': 'string',
                'latitude': 'float64',
                'longitude': 'float64',
                'timezone_offset': 'int32'
            }
            
            self.generated_trajectory = pd.DataFrame(trajectory_data).astype(dtype_map)
            return self.generated_trajectory
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Error processing trajectory data: {str(e)}")

class Discriminator:
    def __init__(self):
        self.score = 0.0
        self.feedback = []
    
    def evaluate_trajectory(self, generated: pd.DataFrame, real: pd.DataFrame) -> Tuple[float, List[str]]:
        prompt = f"""
        Evaluate the following generated trajectory (one day) against the real trajectory (multiple days). When evaluating, consider the data volume difference and focus on pattern matching rather than absolute quantities.
        
        Generated Trajectory:
        {generated.to_string()}
        
        Real Trajectory:
        {real.to_string()}
        
        Please evaluate based on these criteria:
        1. Temporal patterns (0-1 score):
        - Do daily activity timings match typical patterns in real data?
        - Are visit durations proportionally similar?
        
        2. Venue type frequency (0-1 score):
        - Are venue type proportions comparable to daily averages from real data?
        - Is the venue diversity reasonable for a single day?
        
        3. Geographical distribution (0-1 score):
        - Does the spatial coverage align with common areas in real data?
        - Are distances between venues realistic?
        
        4. Venue transition logic (0-1 score):
        - Are venue sequences logical for daily activities?
        - Do transitions follow common patterns seen in real data?
        
        5. Stay duration patterns (0-1 score):
        - Are visit durations appropriate for venue types?
        - Do durations match typical daily patterns?
        
        Return your evaluation in the following JSON format:
        {{
            "overall_score": <average of all scores>,
            "feedback": ["specific feedback point 1", "specific feedback point 2", ... limited 100 words]
        }}
        
        Note: Only include feedback points if overall_score < 0.8
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            # 清理响应内容，移除可能的markdown标记
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            evaluation = json.loads(content)
            # print(evaluation)
            self.score = evaluation['overall_score']
            self.feedback = evaluation.get('feedback', [])
            return self.score, self.feedback
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            raise ValueError(f"无法解析评估结果: {str(e)}")

def main():
    # Load sample data
    data = pd.read_csv('sample_data.csv')
    
    # Initialize components
    profiler = Profiler()
    generator = Generator()
    discriminator = Discriminator()
    
    # Analyze profiles
    long_term_profile = profiler.analyze_long_term_profile(data)
    short_term_pattern = profiler.analyze_short_term_pattern(data)
    
    # Generate trajectory
    generated_trajectory = generator.generate_trajectory(long_term_profile, short_term_pattern, data, feedback_prompt=None)
    
    # Evaluate and regenerate until score is satisfactory
    max_attempts = 3 # 设置最大尝试次数以防止无限循环
    attempt = 0
    while attempt < max_attempts:
        score, feedback = discriminator.evaluate_trajectory(generated_trajectory, data)
        print(f"Attempt {attempt + 1}, Score: {score}")
        
        if score >= 0.8:
            break
            
        print("Regenerating trajectory with feedback...")
        feedback_prompt = f"Previous feedback: {feedback}"
        generated_trajectory = generator.generate_trajectory(
            long_term_profile, 
            short_term_pattern, 
            data,
            feedback_prompt
        )
        attempt += 1
    
    # Save results
    generated_trajectory.to_csv('generated_trajectory.csv', index=False)
    print(f"Final score: {score}")
    print("Results saved to generated_trajectory.csv")

if __name__ == "__main__":
    main() 