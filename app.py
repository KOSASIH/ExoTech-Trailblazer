import numpy as np
import cv2

def analyze_satellite_imagery(image):
    # Preprocess the image (e.g., resize, enhance contrast, etc.)
    preprocessed_image = preprocess_image(image)
    
    # Apply computer vision techniques to detect suitable terrain features
    detected_features = detect_terrain_features(preprocessed_image)
    
    # Analyze the detected features and generate analysis results
    analysis_results = analyze_features(detected_features)
    
    # Generate a markdown report with recommended landing sites and analysis results
    markdown_report = generate_markdown_report(analysis_results)
    
    return markdown_report

def preprocess_image(image):
    # Perform image preprocessing tasks, such as resizing, enhancing contrast, etc.
    # Example:
    resized_image = cv2.resize(image, (500, 500))
    enhanced_image = enhance_contrast(resized_image)
    
    return enhanced_image

def detect_terrain_features(image):
    # Apply computer vision techniques to detect suitable terrain features
    # Example:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def analyze_features(features):
    # Analyze the detected terrain features and generate analysis results
    # Example:
    analysis_results = []
    for contour in features:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        analysis_results.append({
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity
        })
    
    return analysis_results

def generate_markdown_report(analysis_results):
    # Generate a markdown report with recommended landing sites and analysis results
    markdown_report = "## Landing Site Analysis Results\n\n"
    
    for i, result in enumerate(analysis_results):
        markdown_report += f"### Landing Site {i+1}\n"
        markdown_report += f"- Area: {result['area']} square units\n"
        markdown_report += f"- Perimeter: {result['perimeter']} units\n"
        markdown_report += f"- Circularity: {result['circularity']}\n\n"
    
    return markdown_report
