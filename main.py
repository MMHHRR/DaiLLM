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
        3. Income level estimation
        4. Likely occupation
        5. Lifestyle characteristics
        6. Preferred locations and venue types
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
        
        Please identify and provide at least 500 words:
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
    
    def generate_trajectory(self, profile: str, pattern: str, historical_data: pd.DataFrame, 
                          start_date: str = "2025/5/28", num_days: int = 3, 
                          events: List[Dict] = None, feedback_prompt: str = None) -> pd.DataFrame:
        """
        Generate trajectory based on user profile and patterns
        
        Args:
            profile: User profile analysis
            pattern: Mobility pattern analysis
            historical_data: Historical location data
            start_date: Start date for trajectory generation (format: YYYY/MM/DD)
            num_days: Number of days to generate
            events: List of events that may affect mobility, each event is a dict with:
                   {
                       'date': 'YYYY/MM/DD',
                       'type': 'event_type',
                       'description': 'event_description',
                       'impact': 'impact_description'
                   }
            feedback_prompt: Previous feedback to consider
        """
        # Calculate end date
        start_dt = datetime.strptime(start_date, "%Y/%m/%d")
        end_dt = start_dt + timedelta(days=num_days-1)
        
        # Format events information
        events_info = ""
        if events:
            events_info = "\nSpecial Events to Consider:\n"
            for event in events:
                events_info += f"- Date: {event['date']}\n"
                events_info += f"  Type: {event['type']}\n"
                events_info += f"  Description: {event['description']}\n"
                events_info += f"  Impact: {event['impact']}\n"
        
        prompt = f"""
        Generate a {num_days}-day ({start_date}-{end_dt.strftime('%Y/%m/%d')}) trajectory based on:

        User Profile:
        {profile}
        
        Mobility Pattern:
        {pattern}
        
        Historical Location Data:
        {historical_data.to_string()}
        
        {events_info}
        
        Generate a realistic trajectory that includes:
        1. Venue categories
        2. Timestamps
        3. Latitude and longitude coordinates (Refer to Historical Location)
        
        Return the trajectory in the following JSON format:
        {{
            "trajectory": [
                {{
                    "timestamp": "YYYY-MM-DD HH:MM:SS",
                    "venue_category": "string",
                    "latitude": float,
                    "longitude": float
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
    
    def evaluate_trajectory(self, generated: pd.DataFrame, real: pd.DataFrame, num_days: int = 3) -> Tuple[float, List[str]]:
        """
        Evaluate the generated trajectory against the real trajectory
        
        Args:
            generated: Generated trajectory DataFrame
            real: Real trajectory DataFrame for reference
            num_days: Number of days in the generated trajectory
        """
        # Calculate the number of days in the generated trajectory
        generated['date'] = pd.to_datetime(generated['timestamp']).dt.date
        unique_days = len(generated['date'].unique())
        
        prompt = f"""
        Evaluate the following generated trajectory ({num_days} days) against the real trajectory (multiple days). 
        The goal is to assess if this {num_days}-day trajectory could be a realistic part of the longer-term pattern.
        
        Generated Trajectory ({num_days} days):
        {generated.to_string()}
        
        Real Trajectory (Reference data):
        {real.to_string()}
        
        Please evaluate based on these criteria, considering the {num_days}-day vs multi-day context:
        1. Temporal patterns (0.00-1.00 score):
        - Does each day follow common daily rhythm (morning, noon, evening activities)?
        - Award full points if timing matches ANY typical day pattern in real data
        - Consider the impact of special events on daily patterns
        
        2. Venue type frequency (0.00-1.00 score):
        - Compare to the average daily venue type distribution
        - Award full points if proportions are within reasonable daily variation
        - Don't penalize for missing venue types that aren't visited every day
        - Consider how special events might affect venue type preferences
        
        3. Geographical distribution (0.00-1.00 score):
        - Focus on travel distances and area coverage for {num_days} days
        - Award full points if locations fall within common activity zones
        - Don't penalize for not covering all possible areas in one day
        - Consider how special events might affect travel distances
        
        4. Venue transition logic (0.00-1.00 score):
        - Evaluate if each transition makes sense in sequence
        - Award full points for logical daily flow (e.g., home->work->restaurant->home)
        - Consider common daily patterns rather than weekly variety
        - Account for how special events might alter normal transition patterns
        
        5. Stay duration patterns (0.00-1.00 score):
        - Compare durations to typical {num_days}-day patterns
        - Award full points if durations match common venue-specific stays
        - Consider peak/off-peak timing appropriateness
        - Account for how special events might affect stay durations
        
        Return your evaluation in the following JSON format:
        {{
            "overall_score": <average of all scores>,
            "feedback": ["specific suggestion point 1", "specific suggestion point 2", ... limited 100 words]
        }}

        NOTE: Only include feedback if overall_score < 0.85
        """
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL_D,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content = response.choices[0].message.content.strip()
            
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
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
        
    def sample_user_data(self, user_data: pd.DataFrame, max_days: int = 7, min_points: int = 100) -> pd.DataFrame:
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
            if len(sampled_data) < min_points:
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
        
    def process_user(self, user_id: int, user_data: pd.DataFrame, 
                    start_date: str = "2025/5/28", num_days: int = 3,
                    events: List[Dict] = None) -> Dict:
        """Process trajectory generation for a single user
        
        Args:
            user_id: User ID
            user_data: User's historical trajectory data
            start_date: Start date for trajectory generation (format: YYYY/MM/DD)
            num_days: Number of days to generate
            events: List of events that may affect mobility
        """
        try:
            # data sampling
            sampled_data_df = self.sample_user_data(user_data)

            POI_data = sampled_data_df[['venueCategory', 'latitude', 'longitude']].copy()
            sampled_data = sampled_data_df[['userId', 'venueCategory', 'utcTimestamp']].copy()
            
            profiler = Profiler()
            generator = Generator()
            discriminator = Discriminator()
            
            # Analyze user characteristics
            long_term_profile = profiler.analyze_long_term_profile(sampled_data)
            short_term_pattern = profiler.analyze_short_term_pattern(sampled_data)
            
            # Generate and evaluate trajectory
            max_attempts = 3
            attempt = 0
            best_score = 0.0
            best_trajectory = None
            feedback_history = []
            
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
                    start_date=start_date,
                    num_days=num_days,
                    events=events,
                    feedback_prompt=feedback_prompt
                )
                
                score, feedback = discriminator.evaluate_trajectory(
                    current_trajectory, 
                    sampled_data,
                    num_days=num_days
                )
                if score > best_score:
                    best_score = score
                    best_trajectory = current_trajectory

                feedback_history.append({
                    'userId': user_id,
                    'attempt': attempt,
                    'score': score,
                    'feedback': feedback
                })

                if score >= 0.85:
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
                user_feedback = pd.DataFrame(feedback_history)
                
                # Save to temporary files
                user_results.to_csv(self.checkpoint_dir / f'user_{user_id}_trajectories.csv', index=False)
                user_scores.to_csv(self.checkpoint_dir / f'user_{user_id}_scores.csv', index=False)
                user_feedback.to_csv(self.checkpoint_dir / f'user_{user_id}_feedback.csv', index=False)
                
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
            trajectories.to_csv(self.output_dir / 'generated_trajectories_event_TKY.csv', index=False)
        
        # Merge score data
        score_files = list(self.checkpoint_dir.glob('*_scores.csv'))
        if score_files:
            scores = pd.concat([pd.read_csv(f) for f in score_files])
            scores.to_csv(self.output_dir / 'generation_scores_event_TKY.csv', index=False)
        
        # Merge feedback data
        feedback_files = list(self.checkpoint_dir.glob('*_feedback.csv'))
        if feedback_files:
            feedback = pd.concat([pd.read_csv(f) for f in feedback_files])
            feedback.to_csv(self.output_dir / 'generation_feedback_event_TKY.csv', index=False)

    def get_processed_user_ids(self) -> set:
        """Get the set of user IDs that have already been processed"""
        processed_ids = set()
        # Check trajectory files
        trajectory_files = list(self.checkpoint_dir.glob('*_trajectories.csv'))
        for file in trajectory_files:
            try:
                df = pd.read_csv(file)
                if 'userId' in df.columns:
                    processed_ids.update(df['userId'].unique())
            except Exception as e:
                logging.warning(f"Error reading file {file}: {str(e)}")
        return processed_ids

def main():
    # Create necessary directories
    checkpoint_dir = 'checkpoints'
    output_dir = 'output'
    Path(checkpoint_dir).mkdir(exist_ok=True)
    Path(output_dir).mkdir(exist_ok=True)
    
    logging.info("Loading data...")
    data = pd.read_csv(r'D:\A_Research\A_doing_research\20250526_LLM_causal_inference\dataset\dataset_TSMC2014_TKY.csv')
    data = data.drop(['venueId', 'venueCategoryId', 'timezoneOffset'], axis=1)
    data['utcTimestamp'] = pd.to_datetime(data['utcTimestamp'])
    
    processor = TrajectoryProcessor(checkpoint_dir, output_dir)
    
    # Get already processed user IDs
    processed_ids = processor.get_processed_user_ids()
    logging.info(f"Found {len(processed_ids)} already processed users")
    
    # Filter out unprocessed users
    unique_users = data['userId'].unique()
    remaining_users = [uid for uid in unique_users if uid not in processed_ids]
    logging.info(f"Found {len(remaining_users)} users remaining to process")
    
    if not remaining_users:
        logging.info("All users have been processed!")
        return
    
    # Set up parallel processing
    num_processes = mp.cpu_count() - 1  # Reserve one CPU core
    logging.info(f"Using {num_processes} processes for parallel processing")
    
    # Define simulation parameters
    start_date = "2025/5/19"
    num_days = 5
    
    # Define events that may affect mobility
    # events = ['']
    events = [
        {
            'date': '2025/5/20',
            'type': 'weather',
            'description': 'Typhoon approaching',
            'impact': 'Strong typhoon expected to affect the region for 1-2 days. Heavy typhoon may limit outdoor activities and affect transportation. People may prefer indoor venues and stay closer to home.'
        }
    ]
    
    # Process users in batches
    batch_size = 50
    total_batches = (len(remaining_users) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(remaining_users))
        batch_users = remaining_users[start_idx:end_idx]
        
        logging.info(f"Processing batch {batch_idx + 1}/{total_batches} (users {start_idx}-{end_idx})")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for user_id in batch_users:
                user_data = data[data['userId'] == user_id].copy()
                future = executor.submit(
                    processor.process_user, 
                    user_id, 
                    user_data,
                    start_date=start_date,
                    num_days=num_days,
                    events=events
                )
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