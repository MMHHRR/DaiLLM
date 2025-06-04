import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import json
import multiprocessing as mp
from pathlib import Path
import time
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trajectory_generation.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
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
            model=config.LLM_MODEL_P,
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
            model=config.LLM_MODEL_P,
            messages=[{"role": "user", "content": prompt}]
        )
        
        self.short_term_pattern = response.choices[0].message.content
        return self.short_term_pattern

class Generator:
    def __init__(self):
        self.generated_trajectory = None
    
    def generate_trajectory(self, profile: str, pattern: str, historical_data: pd.DataFrame, feedback_prompt: str = None) -> pd.DataFrame:
        prompt = f"""
        Generate a one-day (2025/5/28) trajectory based on:
        
        User Profile:
        {profile}
        
        Mobility Pattern:
        {pattern}
        
        Historical Location Data:
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
            model=config.LLM_MODEL_G,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content = response.choices[0].message.content.strip()
            # 清理可能的 markdown 代码块标记
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # 尝试找到第一个 { 和最后一个 } 之间的内容
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                content = content[start_idx:end_idx]
            
            # 检查返回的内容是否已经是正确的JSON格式
            try:
                data = json.loads(content)
                if 'trajectory' in data:
                    trajectory_data = data['trajectory']
                else:
                    # 如果返回的是轨迹点数组，将其包装在trajectory字段中
                    trajectory_data = json.loads(f'{{"trajectory": {content}}}')['trajectory']
            except json.JSONDecodeError:
                # 如果解析失败，尝试将内容包装在trajectory字段中
                trajectory_data = json.loads(f'{{"trajectory": [{content}]}}')['trajectory']
            
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
            logging.error(f"Error processing trajectory data: {str(e)}")
            logging.error(f"Raw response content: {content}")
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
        1. Temporal patterns (0.00-1.00 score):
        - Does the day follow common daily rhythm (morning, noon, evening activities)?
        - Award full points if timing matches ANY typical day pattern in real data
        
        2. Venue type frequency (0.00-1.00 score):
        - Compare to the average daily venue type distribution
        - Award full points if proportions are within reasonable daily variation
        - Don't penalize for missing venue types that aren't visited every day
        
        3. Geographical distribution (0.00-1.00 score):
        - Focus on travel distances and area coverage for a single day
        - Award full points if locations fall within common activity zones
        - Don't penalize for not covering all possible areas in one day
        
        4. Venue transition logic (0.00-1.00 score):
        - Evaluate if each transition makes sense in sequence
        - Award full points for logical daily flow (e.g., home->work->restaurant->home)
        - Consider common daily patterns rather than weekly variety
        
        5. Stay duration patterns (0.00-1.00 score):
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
            model=config.LLM_MODEL_D,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content = response.choices[0].message.content.strip()
            # 清理可能的 markdown 代码块标记
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # 尝试找到第一个 { 和最后一个 } 之间的内容
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                content = content[start_idx:end_idx]
            
            evaluation = json.loads(content)
            self.score = evaluation['overall_score']
            self.feedback = evaluation.get('feedback', [])
            return self.score, self.feedback
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            print(f"Problematic content: {content}")
            raise ValueError(f"Failed to parse evaluation result: {str(e)}")

class TrajectoryProcessor:
    def __init__(self, checkpoint_dir: str, output_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def sample_user_data(self, user_data: pd.DataFrame, max_days: int = 7, min_points: int = 200) -> pd.DataFrame:
        """Sample user data to ensure it doesn't exceed model's token limit while maintaining sufficient data points
        
        Args:
            user_data: DataFrame containing user trajectory data
            max_days: Maximum number of days to sample
            min_points: Minimum number of data points required
        """
        if not pd.api.types.is_datetime64_any_dtype(user_data['utcTimestamp']):
            user_data['utcTimestamp'] = pd.to_datetime(user_data['utcTimestamp'])
        
        user_data['date'] = user_data['utcTimestamp'].dt.date
        unique_dates = sorted(user_data['date'].unique())
        daily_counts = user_data.groupby('date').size()
        
        # If number of days exceeds max_days, prioritize days with more data points
        if len(unique_dates) > max_days:
            sorted_dates = daily_counts.sort_values(ascending=False).index
            selected_dates = sorted_dates[:max_days]
            sampled_data = user_data[user_data['date'].isin(selected_dates)].copy()
            if len(sampled_data) < 50:
                sampled_data = user_data.copy()
        else:
            sampled_data = user_data.copy()
        
        # If data points exceed minimum requirement, perform uniform sampling
        if len(sampled_data) > min_points:
            sampled_data = sampled_data.sort_values('utcTimestamp')
            step = len(sampled_data) // min_points
            sampled_data = sampled_data.iloc[::step]
            if len(sampled_data) > min_points:
                sampled_data = sampled_data.iloc[:min_points]
        
        sampled_data = sampled_data.drop('date', axis=1)
        return sampled_data
        
    def process_user(self, user_id: int, user_data: pd.DataFrame) -> Dict:
        """Process trajectory generation for a single user"""
        try:
            # data sampling
            sampled_data = self.sample_user_data(user_data)
            print(len(sampled_data))
            # sampled_data = user_data
            
            profiler = Profiler()
            generator = Generator()
            discriminator = Discriminator()
            
            POI_data = sampled_data[['venueCategory', 'latitude', 'longitude']].copy()
            
            # Analyze user characteristics
            long_term_profile = profiler.analyze_long_term_profile(sampled_data)
            short_term_pattern = profiler.analyze_short_term_pattern(sampled_data)
            
            # Generate and evaluate trajectory
            max_attempts = 3
            attempt = 0
            best_score = 0.0
            best_trajectory = None
            
            while attempt < max_attempts:
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
                    POI_data,
                    feedback_prompt=feedback_prompt
                )
                
                score, feedback = discriminator.evaluate_trajectory(current_trajectory, sampled_data)
                
                if score > best_score:
                    best_score = score
                    best_trajectory = current_trajectory
                
                if score >= 0.8:
                    break
                    
                attempt += 1
            
            if best_trajectory is not None:
                best_trajectory['userId'] = user_id
                best_trajectory = best_trajectory.sort_values('timestamp')
                
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
                
                # Save individual user results
                user_results = pd.DataFrame(od_data)
                user_scores = pd.DataFrame([{'userId': user_id, 'score': best_score}])
                
                # Save to temporary files
                user_results.to_csv(self.checkpoint_dir / f'user_{user_id}_trajectories.csv', index=False)
                user_scores.to_csv(self.checkpoint_dir / f'user_{user_id}_scores.csv', index=False)
                
                return {
                    'user_id': user_id,
                    'status': 'success',
                    'score': best_score,
                    'num_trajectories': len(od_data)
                }
            
            return {
                'user_id': user_id,
                'status': 'failed',
                'score': 0.0,
                'num_trajectories': 0
            }
            
        except Exception as e:
            logging.error(f"Error processing user {user_id}: {str(e)}")
            return {
                'user_id': user_id,
                'status': 'error',
                'error': str(e),
                'score': 0.0,
                'num_trajectories': 0
            }
    
    def merge_results(self):
        """Merge all temporary result files"""
        # Merge trajectory data
        trajectory_files = list(self.checkpoint_dir.glob('*_trajectories.csv'))
        if trajectory_files:
            trajectories = pd.concat([pd.read_csv(f) for f in trajectory_files])
            trajectories.to_csv(self.output_dir / 'generated_trajectories2.csv', index=False)
        
        # Merge score data
        score_files = list(self.checkpoint_dir.glob('*_scores.csv'))
        if score_files:
            scores = pd.concat([pd.read_csv(f) for f in score_files])
            scores.to_csv(self.output_dir / 'generation_scores2.csv', index=False)

def main():
    # Create necessary directories
    checkpoint_dir = 'checkpoints'
    output_dir = 'output'
    Path(checkpoint_dir).mkdir(exist_ok=True)
    Path(output_dir).mkdir(exist_ok=True)
    
    logging.info("Loading data...")
    data = pd.read_csv(r'D:\A_Research\A_doing_research\20250526_LLM_causal_inference\dataset\dataset_TSMC2014_NYC.csv')
    data = data.drop(['venueId', 'venueCategoryId'], axis=1)
    data['utcTimestamp'] = pd.to_datetime(data['utcTimestamp'])
    
    unique_users = data['userId'].unique()
    logging.info(f"Found {len(unique_users)} users")
    
    # Set up parallel processing
    num_processes = mp.cpu_count() - 1  # Reserve one CPU core
    logging.info(f"Using {num_processes} processes for parallel processing")
    
    # Process users in batches
    batch_size = 50
    total_batches = (len(unique_users) + batch_size - 1) // batch_size
    
    processor = TrajectoryProcessor(checkpoint_dir, output_dir)
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(unique_users))
        batch_users = unique_users[start_idx:end_idx]
        
        logging.info(f"Processing batch {batch_idx + 1}/{total_batches} (users {start_idx}-{end_idx})")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for user_id in batch_users:
                user_data = data[data['userId'] == user_id].copy()
                future = executor.submit(processor.process_user, user_id, user_data)
                futures.append(future)
            
            # Use tqdm to show progress
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        logging.info(f"Successfully processed user {result['user_id']} with score {result['score']:.2f}")
                    else:
                        logging.warning(f"Failed to process user {result['user_id']}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logging.error(f"Error in future: {str(e)}")
        
        # Merge results after each batch
        processor.merge_results()
        logging.info(f"Completed batch {batch_idx + 1}/{total_batches}")
    
    # Final merge of all results
    processor.merge_results()
    logging.info("Generation completed!")
    logging.info(f"Results saved to {output_dir}/generated_trajectories.csv")
    logging.info(f"Scores saved to {output_dir}/generation_scores.csv")

if __name__ == "__main__":
    main() 