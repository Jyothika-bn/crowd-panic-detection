import numpy as np
from datetime import datetime
import json

class PanicScoreCalculator:
    def __init__(self):
        self.score_history = []
        self.max_history = 10
        
        # Configurable weights for different features
        self.weights = {
            'motion': 0.4,
            'pose': 0.3,
            'emotion': 0.3
        }
        
        # Thresholds for different alert levels
        self.thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.65,
            'critical': 0.8
        }
        
        # Synergy effects configuration
        self.synergy_config = {
            'motion_emotion_threshold': 0.6,
            'motion_emotion_boost': 0.2,
            'pose_motion_threshold': 0.7,
            'pose_motion_boost': 0.15,
            'triple_high_threshold': 0.6,
            'triple_high_boost': 0.25
        }

    def compute_base_score(self, motion, pose, emotion):
        """Compute base panic score using weighted combination"""
        base_score = (
            self.weights['motion'] * motion +
            self.weights['pose'] * pose +
            self.weights['emotion'] * emotion
        )
        return base_score

    def apply_synergy_effects(self, motion, pose, emotion, base_score):
        """Apply non-linear synergy effects between features"""
        synergy_boost = 0.0
        
        # Motion + Emotion synergy (panic often shows both high movement and fear)
        if motion > self.synergy_config['motion_emotion_threshold'] and \
           emotion > self.synergy_config['motion_emotion_threshold']:
            synergy_boost += self.synergy_config['motion_emotion_boost']
        
        # Pose + Motion synergy (abnormal postures with high movement)
        if pose > self.synergy_config['pose_motion_threshold'] and \
           motion > self.synergy_config['pose_motion_threshold']:
            synergy_boost += self.synergy_config['pose_motion_boost']
        
        # Triple high synergy (all features indicating panic)
        if all(score > self.synergy_config['triple_high_threshold'] 
               for score in [motion, pose, emotion]):
            synergy_boost += self.synergy_config['triple_high_boost']
        
        return min(base_score + synergy_boost, 1.0)

    def apply_temporal_smoothing(self, current_score):
        """Apply temporal smoothing to reduce noise and false alarms"""
        self.score_history.append(current_score)
        
        # Maintain history size
        if len(self.score_history) > self.max_history:
            self.score_history.pop(0)
        
        if len(self.score_history) < 3:
            return current_score
        
        # Calculate weighted moving average (recent scores have higher weight)
        weights = np.linspace(0.5, 1.0, len(self.score_history))
        weights = weights / np.sum(weights)
        
        smoothed_score = np.average(self.score_history, weights=weights)
        
        # Trend analysis - if scores are consistently increasing, boost current score
        if len(self.score_history) >= 5:
            recent_trend = np.polyfit(range(5), self.score_history[-5:], 1)[0]
            if recent_trend > 0.05:  # Positive trend
                smoothed_score = min(smoothed_score + 0.05, 1.0)
        
        return smoothed_score

    def get_alert_level(self, panic_score):
        """Determine alert level based on panic score"""
        if panic_score >= self.thresholds['critical']:
            return 'CRITICAL'
        elif panic_score >= self.thresholds['high']:
            return 'HIGH'
        elif panic_score >= self.thresholds['medium']:
            return 'MEDIUM'
        elif panic_score >= self.thresholds['low']:
            return 'LOW'
        else:
            return 'NORMAL'

    def get_confidence_level(self, motion, pose, emotion, panic_score):
        """Calculate confidence level of the panic detection"""
        # Base confidence from feature consistency
        feature_scores = [motion, pose, emotion]
        feature_std = np.std(feature_scores)
        
        # Lower standard deviation means more consistent features = higher confidence
        consistency_confidence = max(0, 1.0 - feature_std)
        
        # Confidence from score magnitude
        magnitude_confidence = min(panic_score / 0.8, 1.0)
        
        # Confidence from temporal stability
        temporal_confidence = 1.0
        if len(self.score_history) >= 3:
            recent_std = np.std(self.score_history[-3:])
            temporal_confidence = max(0, 1.0 - recent_std * 2)
        
        # Combined confidence
        overall_confidence = (
            0.4 * consistency_confidence +
            0.3 * magnitude_confidence +
            0.3 * temporal_confidence
        )
        
        return min(overall_confidence, 1.0)

    def analyze_feature_contributions(self, motion, pose, emotion):
        """Analyze individual feature contributions to panic detection"""
        contributions = {
            'motion': motion * self.weights['motion'],
            'pose': pose * self.weights['pose'],
            'emotion': emotion * self.weights['emotion']
        }
        
        # Identify dominant feature
        dominant_feature = max(contributions.keys(), key=lambda k: contributions[k])
        
        # Calculate feature balance (how evenly distributed the contributions are)
        contrib_values = list(contributions.values())
        balance_score = 1.0 - np.std(contrib_values) / np.mean(contrib_values) if np.mean(contrib_values) > 0 else 0
        
        return {
            'contributions': contributions,
            'dominant_feature': dominant_feature,
            'balance_score': balance_score
        }

def compute_panic(motion, pose, emotion):
    """
    Advanced Panic Score Computation with Multi-Modal Feature Fusion
    
    Args:
        motion (float): Motion analysis score (0.0 to 1.0)
        pose (float): Pose analysis score (0.0 to 1.0)  
        emotion (float): Emotion analysis score (0.0 to 1.0)
    
    Returns:
        float: Final panic score (0.0 to 1.0)
    """
    # Initialize calculator (in production, this would be a singleton)
    calculator = PanicScoreCalculator()
    
    # Validate inputs
    motion = max(0.0, min(1.0, motion))
    pose = max(0.0, min(1.0, pose))
    emotion = max(0.0, min(1.0, emotion))
    
    # Compute base score
    base_score = calculator.compute_base_score(motion, pose, emotion)
    
    # Apply synergy effects
    synergy_score = calculator.apply_synergy_effects(motion, pose, emotion, base_score)
    
    # Apply temporal smoothing
    final_score = calculator.apply_temporal_smoothing(synergy_score)
    
    return final_score

def compute_panic_detailed(motion, pose, emotion):
    """
    Compute panic score with detailed analysis
    
    Returns:
        dict: Detailed panic analysis including scores, confidence, and metadata
    """
    calculator = PanicScoreCalculator()
    
    # Validate inputs
    motion = max(0.0, min(1.0, motion))
    pose = max(0.0, min(1.0, pose))
    emotion = max(0.0, min(1.0, emotion))
    
    # Compute scores
    base_score = calculator.compute_base_score(motion, pose, emotion)
    synergy_score = calculator.apply_synergy_effects(motion, pose, emotion, base_score)
    final_score = calculator.apply_temporal_smoothing(synergy_score)
    
    # Get additional analysis
    alert_level = calculator.get_alert_level(final_score)
    confidence = calculator.get_confidence_level(motion, pose, emotion, final_score)
    feature_analysis = calculator.analyze_feature_contributions(motion, pose, emotion)
    
    return {
        'panic_score': final_score,
        'base_score': base_score,
        'synergy_boost': synergy_score - base_score,
        'alert_level': alert_level,
        'confidence': confidence,
        'feature_scores': {
            'motion': motion,
            'pose': pose,
            'emotion': emotion
        },
        'feature_contributions': feature_analysis['contributions'],
        'dominant_feature': feature_analysis['dominant_feature'],
        'balance_score': feature_analysis['balance_score'],
        'timestamp': datetime.now().isoformat(),
        'weights_used': calculator.weights.copy(),
        'thresholds': calculator.thresholds.copy()
    }

def update_fusion_weights(motion_weight=0.4, pose_weight=0.3, emotion_weight=0.3):
    """
    Update feature fusion weights (for fine-tuning)
    
    Args:
        motion_weight (float): Weight for motion features
        pose_weight (float): Weight for pose features
        emotion_weight (float): Weight for emotion features
    """
    # Normalize weights to sum to 1.0
    total = motion_weight + pose_weight + emotion_weight
    if total > 0:
        return {
            'motion': motion_weight / total,
            'pose': pose_weight / total,
            'emotion': emotion_weight / total
        }
    else:
        return {'motion': 0.4, 'pose': 0.3, 'emotion': 0.3}

def get_fusion_statistics():
    """Get feature fusion statistics and configuration"""
    calculator = PanicScoreCalculator()
    
    return {
        'algorithm': 'Advanced Multi-Modal Feature Fusion',
        'base_weights': calculator.weights,
        'alert_thresholds': calculator.thresholds,
        'synergy_effects': calculator.synergy_config,
        'temporal_smoothing': {
            'history_length': calculator.max_history,
            'method': 'Weighted Moving Average with Trend Analysis'
        },
        'features': [
            'Non-linear synergy effects',
            'Temporal smoothing',
            'Confidence estimation',
            'Feature contribution analysis',
            'Alert level classification'
        ]
    }
