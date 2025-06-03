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
        
        Please identify, limited 100 words:
        1. Peak activity periods
        2. Key destinations
        3. Daily routes
        4. Temporal patterns
        5. Categorical preferences
        6. Transportation mode
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
        Generate a one-day trajectory based on:
        
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
        Evaluate the following generated trajectory (one day) against the real trajectory (multiple days). The goal is to assess if this single day trajectory could be a realistic part of the longer-term pattern.
        
        Generated Trajectory (1 day):
        {generated.to_string()}
        
        Real Trajectory (Reference data):
        {real.to_string()}
        
        Please evaluate based on these criteria, considering the single day vs multi-day context:
        1. Temporal patterns (0-1 score):
        - Does the day follow common daily rhythm (morning, noon, evening activities)?
        - Award full points if timing matches ANY typical day pattern in real data
        
        2. Venue type frequency (0-1 score):
        - Compare to the average daily venue type distribution
        - Award full points if proportions are within reasonable daily variation
        - Don't penalize for missing venue types that aren't visited every day
        
        3. Geographical distribution (0-1 score):
        - Focus on travel distances and area coverage for a single day
        - Award full points if locations fall within common activity zones
        - Don't penalize for not covering all possible areas in one day
        
        4. Venue transition logic (0-1 score):
        - Evaluate if each transition makes sense in sequence
        - Award full points for logical daily flow (e.g., home->work->restaurant->home)
        - Consider common daily patterns rather than weekly variety
        
        5. Stay duration patterns (0-1 score):
        - Compare durations to typical single-day patterns
        - Award full points if durations match common venue-specific stays
        - Consider peak/off-peak timing appropriateness
        
        Return your evaluation in the following JSON format:
        {{
            "overall_score": <average of all scores>,
            "feedback": ["specific suggestion point 1", "specific suggestion point 2", ... limited 100 words]
        }}
        
        Note: Only include feedback if overall_score < 0.8
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            evaluation = json.loads(content)
            self.score = evaluation['overall_score']
            self.feedback = evaluation.get('feedback', [])
            return self.score, self.feedback
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            raise ValueError(f"Failed to parse evaluation result: {str(e)}")

def main():
    # 1. Data loading and preprocessing
    print("Loading data...")
    data = pd.read_csv(r'D:\A_Research\A_doing_research\20250526_LLM_causal_inference\dataset\dataset_TSMC2014_NYC.csv')
    data = data.drop(['venueId', 'venueCategoryId'], axis=1)
    data['utcTimestamp'] = pd.to_datetime(data['utcTimestamp'])
    
    # 2. Initialize components
    profiler = Profiler()
    generator = Generator()
    discriminator = Discriminator()
    
    # 3. Get all users
    unique_users = data['userId'].unique()
    print(f"Found {len(unique_users)} users")
    
    # 4. Process each user
    all_results = []
    all_scores = []
    
    for idx, user_id in enumerate(unique_users):
        if idx >= 3:
            break
        print(f"\nProcessing user {user_id}...")
        user_data = data[data['userId'] == user_id].copy()
        
        # 4.1 Analyze user features
        long_term_profile = profiler.analyze_long_term_profile(user_data)
        short_term_pattern = profiler.analyze_short_term_pattern(user_data)
        
        # 4.2 Generate and evaluate trajectory
        max_attempts = 3
        attempt = 0
        best_score = 0.0
        best_trajectory = None
        
        while attempt < max_attempts:
            # Generate trajectory
            feedback_prompt = None
            if attempt > 0:
                feedback_prompt = f"""
                Previous generated trajectory: {current_trajectory.to_string()}
                Previous evaluation score: {score}
                Previous feedback to consider: {feedback}
                """
            
            current_trajectory = generator.generate_trajectory(
                long_term_profile, 
                short_term_pattern, 
                user_data,
                feedback_prompt=feedback_prompt
            )
            
            # Evaluate trajectory
            score, feedback = discriminator.evaluate_trajectory(current_trajectory, user_data)
            print(f"Attempt {attempt + 1}, Score: {score}")
            
            # Update best result if current score is higher
            if score > best_score:
                best_score = score
                best_trajectory = current_trajectory
            
            if score >= 0.8:
                break
                
            attempt += 1
        
        # 4.3 Convert to OD data and save results
        if best_trajectory is not None:
            best_trajectory['userId'] = user_id
            
            # Sort by timestamp
            best_trajectory = best_trajectory.sort_values('timestamp')
            
            # Create OD pairs
            od_data = []
            for i in range(len(best_trajectory) - 1):
                current = best_trajectory.iloc[i]
                next_point = best_trajectory.iloc[i + 1]
                
                od_pair = {
                    'userId': user_id,
                    'timestamp': current['timestamp'],
                    'origin_lat': current['latitude'],
                    'origin_lon': current['longitude'],
                    'origin_category': current['venue_category'],
                    'destination_lat': next_point['latitude'],
                    'destination_lon': next_point['longitude'],
                    'destination_category': next_point['venue_category']
                }
                od_data.append(od_pair)
            
            all_results.extend(od_data)
            all_scores.append({'userId': user_id, 'score': best_score})
    
    # 5. Save final results
    if all_results:
        final_results = pd.DataFrame(all_results)
        scores_df = pd.DataFrame(all_scores)
        
        final_results.to_csv('generated_trajectories.csv', index=False)
        scores_df.to_csv('generation_scores.csv', index=False)
        
        print("\nGeneration completed!")
        print(f"Results saved to generated_trajectories.csv")
        print(f"Scores saved to generation_scores.csv")
    else:
        print("No trajectories were successfully generated")

if __name__ == "__main__":
    main() 