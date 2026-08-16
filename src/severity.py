def estimate_severity(box_area, image_area, confidence):
    """
    Estimates the visual severity of a pothole based on its relative size in the image.
    NOTE: This is 'Estimated Visual Severity' and NOT actual physical depth.
    
    Args:
        box_area (float): Area of the bounding box (width * height in pixels).
        image_area (float): Total area of the image (width * height in pixels).
        confidence (float): The model's confidence score (0.0 to 1.0).
        
    Returns:
        str: "LOW", "MEDIUM", or "HIGH"
    """
    # Calculate how much of the image the pothole takes up (percentage)
    relative_area = (box_area / image_area) * 100

    # If confidence is very low, it might be a shallow/unclear pothole
    # We combine relative area size with confidence to gauge severity.
    
    if relative_area > 5.0:  # Takes up more than 5% of the frame (Very Large)
        return "HIGH"
    elif relative_area > 1.5: # Takes up between 1.5% and 5% (Medium)
        return "MEDIUM"
    else:                    # Less than 1.5% of the frame (Small)
        return "LOW"

def calculate_road_condition(pothole_count, severities):
    """
    Calculates an 'AI Estimated Road Condition' score based on detections.
    """
    if pothole_count == 0:
        return "GOOD"
        
    score = 0
    for sev in severities:
        if sev == "HIGH":
            score += 3
        elif sev == "MEDIUM":
            score += 2
        else:
            score += 1
            
    # Normalize score based on pothole count
    avg_score = score / pothole_count
    
    if pothole_count >= 5 or avg_score >= 2.5:
        return "CRITICAL"
    elif pothole_count >= 3 or avg_score >= 1.5:
        return "POOR"
    else:
        return "MODERATE"
