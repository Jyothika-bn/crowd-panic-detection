import cv2
import numpy as np

# Global variables for optical flow
prev_gray = None
flow_history = []
MAX_FLOW_HISTORY = 10

def calculate_optical_flow(current_gray, prev_gray):
    """Calculate dense optical flow using Farneback method"""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None,
        pyr_scale=0.5,      # Pyramid scale
        levels=3,           # Number of pyramid levels
        winsize=15,         # Window size
        iterations=3,       # Iterations at each pyramid level
        poly_n=5,          # Neighborhood size
        poly_sigma=1.2,    # Gaussian sigma
        flags=0
    )
    return flow

def analyze_flow_patterns(flow):
    """Analyze optical flow patterns for panic indicators"""
    if flow is None:
        return {
            'magnitude': 0.0,
            'direction_chaos': 0.0,
            'velocity_variance': 0.0,
            'flow_density': 0.0
        }
    
    # Calculate flow magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 1. Average flow magnitude (speed)
    avg_magnitude = float(np.mean(magnitude))
    
    # 2. Direction chaos (standard deviation of flow directions)
    # Higher chaos indicates panic/confusion
    significant_flow_mask = magnitude > 1.0  # Only consider significant motion
    if np.any(significant_flow_mask):
        flow_angles = angle[significant_flow_mask]
        direction_chaos = float(np.std(flow_angles))
    else:
        direction_chaos = 0.0
    
    # 3. Velocity variance (inconsistent speeds indicate panic)
    velocity_variance = float(np.var(magnitude))
    
    # 4. Flow density (percentage of frame with significant motion)
    flow_density = float(np.sum(significant_flow_mask) / magnitude.size)
    
    return {
        'magnitude': avg_magnitude,
        'direction_chaos': direction_chaos,
        'velocity_variance': velocity_variance,
        'flow_density': flow_density
    }

def detect_crowd_behaviors(flow_patterns, person_count):
    """Detect specific crowd behaviors from flow patterns"""
    behaviors = {
        'normal_flow': False,
        'rushing': False,
        'chaotic_movement': False,
        'stampede_risk': False,
        'convergence': False,
        'divergence': False
    }
    
    magnitude = flow_patterns['magnitude']
    chaos = flow_patterns['direction_chaos']
    variance = flow_patterns['velocity_variance']
    density = flow_patterns['flow_density']
    
    # Normal flow: low magnitude, low chaos
    if magnitude < 3.0 and chaos < 1.0:
        behaviors['normal_flow'] = True
    
    # Rushing: high magnitude, moderate chaos
    if magnitude > 8.0 and chaos < 2.0:
        behaviors['rushing'] = True
    
    # Chaotic movement: high chaos regardless of magnitude
    if chaos > 2.5:
        behaviors['chaotic_movement'] = True
    
    # Stampede risk: high magnitude + high chaos + high density
    if magnitude > 10.0 and chaos > 2.0 and density > 0.3:
        behaviors['stampede_risk'] = True
    
    # Convergence/Divergence analysis would require more complex flow field analysis
    # For now, we use simplified heuristics
    if density > 0.5 and variance > 20.0:
        if magnitude > 6.0:
            behaviors['divergence'] = True  # People spreading out quickly
        else:
            behaviors['convergence'] = True  # People gathering
    
    return behaviors

def calculate_panic_indicators(flow_patterns, behaviors, person_count):
    """Calculate panic indicators from flow analysis"""
    panic_score = 0.0
    
    # Base score from flow magnitude (normalized)
    magnitude_score = min(flow_patterns['magnitude'] / 15.0, 1.0)
    panic_score += magnitude_score * 0.3
    
    # Chaos contribution (high chaos = panic)
    chaos_score = min(flow_patterns['direction_chaos'] / 3.14, 1.0)  # Normalize by π
    panic_score += chaos_score * 0.4
    
    # Density contribution (crowded areas more prone to panic)
    density_score = min(flow_patterns['flow_density'], 1.0)
    panic_score += density_score * 0.2
    
    # Behavior-based scoring
    if behaviors['stampede_risk']:
        panic_score += 0.5
    elif behaviors['chaotic_movement']:
        panic_score += 0.3
    elif behaviors['rushing']:
        panic_score += 0.2
    
    # Crowd size amplification
    if person_count > 5:
        crowd_amplifier = min((person_count - 5) / 15.0, 0.3)  # Max 30% boost
        panic_score += crowd_amplifier
    
    return min(panic_score, 1.0)

def motion_score(frame, person_count=0):
    """
    Advanced Motion Analysis for Crowd Panic Detection
    
    Args:
        frame: Input video frame
        person_count: Number of people detected in frame
    
    Returns:
        float: Motion-based panic score (0.0 to 1.0)
    """
    global prev_gray, flow_history
    
    if frame is None:
        return 0.0
    
    try:
        # Convert to grayscale
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize on first frame
        if prev_gray is None:
            prev_gray = current_gray.copy()
            return 0.0
        
        # Calculate optical flow
        flow = calculate_optical_flow(current_gray, prev_gray)
        
        # Analyze flow patterns
        flow_patterns = analyze_flow_patterns(flow)
        
        # Detect crowd behaviors
        behaviors = detect_crowd_behaviors(flow_patterns, person_count)
        
        # Calculate panic indicators
        panic_score = calculate_panic_indicators(flow_patterns, behaviors, person_count)
        
        # Update flow history for temporal analysis
        flow_history.append(flow_patterns)
        if len(flow_history) > MAX_FLOW_HISTORY:
            flow_history.pop(0)
        
        # Temporal consistency check
        if len(flow_history) >= 3:
            # Check if panic indicators are consistent over time
            recent_scores = [calculate_panic_indicators(fp, detect_crowd_behaviors(fp, person_count), person_count) 
                           for fp in flow_history[-3:]]
            
            # If consistently high, boost confidence
            if all(score > 0.6 for score in recent_scores):
                panic_score = min(panic_score + 0.1, 1.0)
        
        # Update previous frame
        prev_gray = current_gray.copy()
        
        return panic_score
        
    except Exception as e:
        print(f"Motion analysis error: {e}")
        return 0.0

def draw_optical_flow(frame, flow, step=16):
    """
    Draw optical flow vectors on frame for visualization
    
    Args:
        frame: Input frame
        flow: Optical flow field
        step: Step size for drawing vectors
    
    Returns:
        frame: Frame with flow vectors drawn
    """
    if flow is None:
        return frame
    
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    
    fx, fy = flow[y, x].T
    
    # Create line endpoints
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    # Draw flow vectors
    for (x1, y1), (x2, y2) in lines:
        # Color based on magnitude
        magnitude = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if magnitude > 2:  # Only draw significant motion
            color = (0, int(255 * min(magnitude/10, 1)), 0)  # Green intensity based on speed
            cv2.arrowedLine(frame, (x1, y1), (x2, y2), color, 1, tipLength=0.3)
    
    return frame

def get_motion_statistics():
    """Get motion analysis statistics"""
    return {
        "algorithm": "Farneback Optical Flow",
        "pyramid_levels": 3,
        "window_size": 15,
        "history_length": MAX_FLOW_HISTORY,
        "panic_indicators": [
            "High motion magnitude",
            "Direction chaos",
            "Velocity variance", 
            "Flow density",
            "Temporal consistency"
        ],
        "behaviors_detected": [
            "Normal flow",
            "Rushing",
            "Chaotic movement",
            "Stampede risk",
            "Convergence/Divergence"
        ]
    }

def reset_motion_analysis():
    """Reset motion analysis state"""
    global prev_gray, flow_history
    prev_gray = None
    flow_history = []
