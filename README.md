# Anntate-Depth-Camera-Frames
Creating a Python code framework for annotating ZED X depth files to identify road damage, categorize damage severity, and extract bounding boxes with center points and depth information involves the use of ZED SDK, OpenCV, and annotation tools. Below is an implementation outline:
Dependencies

    ZED SDK: For handling .svo files and depth camera data.
    OpenCV: For image processing and visualization.
    PyTorch/TensorFlow: For integrating pre-trained models to assist in detecting road damage.
    Annotation Library: Such as labelme for creating bounding boxes.
    JSON/YAML: For saving annotations.

1. Install Required Packages

pip install pyzed-python opencv-python labelme numpy matplotlib

2. Code Framework

import numpy as np
import cv2
import json
import os
import pyzed.sl as sl

# Constants for severity levels
SEVERITY_LEVELS = {'B': "Moderate", 'C': "Severe", 'D': "Critical"}

# Initialize ZED camera
def initialize_zed_camera(svo_file):
    """Initialize the ZED camera for reading SVO file."""
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file)
    init_params.coordinate_units = sl.UNIT.METER  # Depth in meters
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        exit(1)
    return zed

# Extract depth and RGB images from ZED
def extract_images(zed):
    """Extract RGB and depth information from ZED camera."""
    runtime_params = sl.RuntimeParameters()
    rgb_image = sl.Mat()
    depth_image = sl.Mat()
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(rgb_image, sl.VIEW.LEFT)  # RGB Image
        zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)  # Depth data
        return rgb_image.get_data(), depth_image.get_data()
    return None, None

# Annotate damage with bounding boxes
def annotate_damage(rgb_frame, annotations):
    """Annotate road damage on the RGB frame."""
    annotated_frame = rgb_frame.copy()
    for anno in annotations:
        bbox = anno['bbox']
        severity = anno['severity']
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        # Draw bounding box
        cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Severity: {SEVERITY_LEVELS[severity]}",
                    (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
    return annotated_frame

# Save annotations
def save_annotations(annotations, output_file):
    """Save annotations as JSON."""
    with open(output_file, "w") as file:
        json.dump(annotations, file, indent=4)

# Main processing loop
def process_svo_file(svo_file, annotations_reference, output_dir):
    """Process SVO file to annotate road damage."""
    zed = initialize_zed_camera(svo_file)
    frame_count = 0

    while True:
        rgb_frame, depth_frame = extract_images(zed)
        if rgb_frame is None:
            break

        # Fetch corresponding annotations (mock example)
        annotations = annotations_reference.get(str(frame_count), [])

        # Annotate image
        annotated_frame = annotate_damage(rgb_frame, annotations)

        # Save annotated image
        cv2.imwrite(os.path.join(output_dir, f"annotated_frame_{frame_count}.png"), annotated_frame)

        # Save depth data for bounding boxes
        for anno in annotations:
            bbox = anno['bbox']
            depth_value = depth_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]].mean()
            anno['depth'] = depth_value

        # Save annotations
        save_annotations(annotations, os.path.join(output_dir, f"annotations_{frame_count}.json"))

        frame_count += 1

    zed.close()
    print("Processing complete.")

# Example usage
if __name__ == "__main__":
    svo_file = "path_to_file.svo"
    annotations_reference = {  # Example mock annotations for reference
        "0": [
            {"bbox": [100, 150, 200, 250], "severity": "D"},
            {"bbox": [300, 400, 350, 450], "severity": "B"}
        ]
    }
    output_dir = "output_annotations"
    os.makedirs(output_dir, exist_ok=True)
    process_svo_file(svo_file, annotations_reference, output_dir)

3. Key Features of the Code

    Bounding Boxes: Drawn using OpenCV based on reference annotations.
    Depth Extraction: The depth information for each bounding box is calculated as the average depth in the region.
    JSON Output: Annotations (bounding box, severity, depth) are saved as a JSON file for further analysis.
    Compatibility: Uses ZED SDK to directly process .svo files and depth data.

Next Steps

    Reference Annotations: Align GoPro-based annotations with ZED SVO frames.
    Fine-Tune Detection: Replace mock annotations with AI-based object detection for automated annotation.
    Integration: Deploy the system on a server for continuous video processing.

Let me know if further clarification or enhancements are required!
