import cv2
import numpy as np

# Initialize lists to store metrics over time for calculating thresholds
brightness_values = []
variance_values = []
edge_density_values = []

# main method for detecting obstruction
def is_obstructed(frame, brightness_thresholdup, brightness_thresholddown, variance_thresholdup, variance_thresholddown, edge_thresholdup, edge_thresholddown):
    # 1. Brightness and variance check
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_frame)
    variance = np.var(gray_frame)
    
    # used to see stats TO BE REMOVED LATER
    print("Brightness and Variance:", avg_brightness, variance)
    brightness_out_of_range = avg_brightness < brightness_thresholddown or avg_brightness > brightness_thresholdup
    variance_out_of_range = variance < variance_thresholddown
    # or variance > variance_thresholdup

    #If brightness or variance change too much we are obstructed
    if brightness_out_of_range or variance_out_of_range:
        print("Brightness or Variance")
        return True  # Likely obstruction due to uniform pixels

    # 2. Edge detection
    # Potential to change how the edges are detected.
    edges = cv2.Canny(gray_frame, 50, 150)
    edge_density = np.sum(edges) / edges.size
    print("Edge Density:", edge_density)
    # If edge density changes too much we are obstructed
    if edge_density < edge_thresholddown or edge_density > edge_density_thresholdup:
        print("Edge")
        return True  # Few edges, likely obstruction

    return False  # No obstruction detected

# These three return the avg stats for the thresholds
def calculate_average_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness

def calculate_variance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    return variance

def calculate_edge_density(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size
    return edge_density

# Function to calculate dynamic thresholds based on collected data
def calculate_dynamic_thresholds(values, deviation_factor):
    """
    Calculate the adaptive threshold based on mean and standard deviation.
    
    Args:
    - values: List of metric values (brightness, variance, etc.).
    - deviation_factor: Factor for how strict the threshold should be. Higher values make it more lenient.
    
    Returns:
    - lower_threshold: Lower bound threshold.
    - upper_threshold: Upper bound threshold (optional, can be used if needed).
    """
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Having a lower and upper threshold isn't bad but may not need it for brightness or variants
    lower_threshold = mean_val - deviation_factor * std_val
    upper_threshold = mean_val + deviation_factor * std_val  # Optional, not used for obstruction detection
    return upper_threshold, lower_threshold

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Capture baseline metrics over N frames for threshold calculation
baseline_frames = 100
frame_count = 0
thresholds_calculated = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Calculate metrics for the current frame
    avg_brightness = calculate_average_brightness(frame)
    variance = calculate_variance(frame)
    edge_density = calculate_edge_density(frame)
    
    # Collect baseline metrics for dynamic threshold calculation
    if frame_count < baseline_frames:
        brightness_values.append(avg_brightness)
        variance_values.append(variance)
        edge_density_values.append(edge_density)
        frame_count += 1
    elif not thresholds_calculated:
        # Once baseline frames are collected, calculate dynamic thresholds
        # these numbers will change the sensitivity of our calculations. May want to make it so the deviation one way is higher than the deviation another
        # ie make the lower threshhold closer to the avg than the higher threshhold.
        brightness_thresholdup, brightness_thresholddown = calculate_dynamic_thresholds(brightness_values, 12)
        variance_thresholdup, variance_thresholddown = calculate_dynamic_thresholds(variance_values, 16)
        edge_density_thresholdup, edge_density_thresholddown = calculate_dynamic_thresholds(edge_density_values, 10)
        
        # Used for testing TO BE REMOVED
        print(f"Dynamic Brightness Threshold: {brightness_thresholddown} to {brightness_thresholdup}")
        print(f"Dynamic Variance Threshold: {variance_thresholddown} to {variance_thresholdup}")
        print(f"Dynamic Edge Density Threshold: {edge_density_thresholddown} to {edge_density_thresholdup}")
        
        thresholds_calculated = True
    
    if thresholds_calculated:
        # Check if the frame is obstructed based on dynamic thresholds
        obstructed = is_obstructed(frame, brightness_thresholdup, brightness_thresholddown, variance_thresholdup, variance_thresholddown, edge_density_thresholdup, edge_density_thresholddown)
    
        # Display status on the frame
        status_text = "Camera Obstructed!" if obstructed else "Camera Clear"
        color = (0, 0, 255) if obstructed else (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, status_text, (10, 30), font, 0.8, color, 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Camera', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
