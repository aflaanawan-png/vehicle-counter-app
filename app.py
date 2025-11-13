import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import pandas as pd
from datetime import datetime
import tempfile
import os

# Page configuration
st.set_page_config(page_title="Vehicle Counter", layout="wide")
st.title("🚗 FHWA Vehicle Counter with YOLOv8")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
skip_frames = st.sidebar.slider("Skip Frames (Speed)", 1, 10, 2)
line_position = st.sidebar.slider("Counting Line Position", 0.0, 1.0, 0.5, 0.05)

# FHWA Vehicle Classes
st.sidebar.header("📊 FHWA Vehicle Classes")
fhwa_classes = {
    1: "Motorcycle",
    2: "Passenger Car",
    3: "Pickup/Van",
    4: "Bus",
    5: "Single Unit Truck (2-axle)",
    6: "Single Unit Truck (3-axle)",
    7: "Single Unit Truck (4+ axle)",
    8: "Single Trailer Truck (3-4 axle)",
    9: "Single Trailer Truck (5-axle)",
    10: "Single Trailer Truck (6+ axle)",
    11: "Multi-Trailer Truck (5 axle)",
    12: "Multi-Trailer Truck (6 axle)",
    13: "Multi-Trailer Truck (7+ axle)"
}

for cls, name in fhwa_classes.items():
    st.sidebar.text(f"Class {cls}: {name}")

# YOLO to FHWA mapping (simplified based on COCO classes)
def map_to_fhwa(yolo_class, bbox_area):
    """Map YOLO class to FHWA vehicle class"""
    # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    if yolo_class == 3:  # motorcycle
        return 1
    elif yolo_class == 2:  # car
        if bbox_area < 5000:
            return 2  # Passenger car
        else:
            return 3  # Pickup/Van
    elif yolo_class == 5:  # bus
        return 4
    elif yolo_class == 7:  # truck
        if bbox_area < 8000:
            return 5  # Small truck
        elif bbox_area < 15000:
            return 8  # Medium truck
        else:
            return 9  # Large truck
    return 2  # Default to passenger car

# Vehicle tracker with improved logic
class VehicleTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.counted = set()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, fhwa_class):
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'class': fhwa_class,
            'crossed': False
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections, line_y):
        """
        detections: list of (centroid, fhwa_class) tuples
        line_y: y-coordinate of counting line
        """
        newly_counted = []
        
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return newly_counted
        
        input_centroids = np.array([d[0] for d in detections])
        input_classes = [d[1] for d in detections]
        
        if len(self.objects) == 0:
            for i, (centroid, fhwa_class) in enumerate(detections):
                self.register(centroid, fhwa_class)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid]['centroid'] for oid in object_ids])
            
            # Calculate distances
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            
            # Match existing objects to new detections
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                old_centroid = self.objects[object_id]['centroid']
                new_centroid = input_centroids[col]
                
                # Update object
                self.objects[object_id]['centroid'] = new_centroid
                self.disappeared[object_id] = 0
                
                # Check if crossed line (only count downward crossing)
                if (object_id not in self.counted and 
                    not self.objects[object_id]['crossed'] and
                    old_centroid[1] < line_y <= new_centroid[1]):
                    
                    self.objects[object_id]['crossed'] = True
                    self.counted.add(object_id)
                    newly_counted.append(self.objects[object_id]['class'])
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle disappeared objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_classes[col])
        
        return newly_counted

# File uploader
uploaded_file = st.file_uploader("📁 Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Display video
    st.video(video_path)
    
    if st.button("▶️ Start Processing", type="primary"):
        try:
            # Load YOLO model
            with st.spinner("🔄 Loading YOLOv8 model..."):
                model = YOLO('yolov8n.pt')  # Using nano model for speed
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate line position
            line_y = int(height * line_position)
            
            # Initialize tracker and counters
            tracker = VehicleTracker(max_disappeared=fps, max_distance=150)
            class_counts = defaultdict(int)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_placeholder = st.empty()
            
            frame_count = 0
            processed_frames = 0
            
            # Create output video
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps//skip_frames, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for speed
                if frame_count % skip_frames != 0:
                    continue
                
                processed_frames += 1
                
                # Run YOLO detection with lower confidence
                results = model(frame, conf=confidence, verbose=False)
                
                # Extract detections
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        # Only detect vehicles: car(2), motorcycle(3), bus(5), truck(7)
                        if cls in [2, 3, 5, 7]:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                            bbox_area = (x2 - x1) * (y2 - y1)
                            fhwa_class = map_to_fhwa(cls, bbox_area)
                            detections.append((centroid, fhwa_class))
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0, 0, 255), -1)
                            cv2.putText(frame, f"Class {fhwa_class}", (int(x1), int(y1)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Update tracker
                newly_counted = tracker.update(detections, line_y)
                for fhwa_class in newly_counted:
                    class_counts[fhwa_class] += 1
                
                # Draw counting line
                cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
                cv2.putText(frame, "COUNTING LINE", (10, line_y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw counts
                y_offset = 30
                total_count = sum(class_counts.values())
                cv2.putText(frame, f"Total: {total_count}", (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Write frame
                out.write(frame)
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing: {frame_count}/{total_frames} frames | Detected: {total_count} vehicles")
                
                # Show frame every 30 processed frames
                if processed_frames % 30 == 0:
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                          channels="RGB", use_container_width=True)
            
            cap.release()
            out.release()
            
            # Show final results
            st.success("✅ Processing Complete!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Vehicle Count Summary")
                total = sum(class_counts.values())
                st.metric("Total Vehicles", total)
                
                # Create DataFrame
                results_data = []
                for cls in range(1, 14):
                    count = class_counts.get(cls, 0)
                    results_data.append({
                        'FHWA Class': cls,
                        'Vehicle Type': fhwa_classes[cls],
                        'Count': count,
                        'Percentage': f"{(count/total*100):.1f}%" if total > 0 else "0%"
                    })
                
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
            
            with col2:
                st.subheader("📈 Distribution")
                chart_data = df[df['Count'] > 0][['Vehicle Type', 'Count']]
                if not chart_data.empty:
                    st.bar_chart(chart_data.set_index('Vehicle Type'))
            
            # Download processed video
            st.subheader("⬇️ Download Results")
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name=f"processed_{uploaded_file.name}",
                    mime="video/mp4"
                )
            
            # Download CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_data = {
                'Time Slot': [datetime.now().strftime("%I:%M %p")],
                **{f'Class {i}': [class_counts.get(i, 0)] for i in range(1, 14)},
                'Total': [total]
            }
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_string,
                file_name=f"vehicle_counts_{timestamp}.csv",
                mime="text/csv"
            )
            
            # Cleanup
            os.unlink(output_path)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    # Cleanup temp file
    os.unlink(video_path)

else:
    st.info("👆 Please upload a video file to begin")
    st.markdown("""
    ### 📋 Instructions:
    1. Upload a traffic video (MP4, AVI, MOV, MKV)
    2. Adjust detection confidence and counting line position
    3. Click "Start Processing" to analyze
    4. Download results and CSV report
    
    ### 🎯 Features:
    - ✅ **No double counting** - Advanced tracking prevents re-counting
    - ✅ **FHWA classification** - Automatic vehicle type detection
    - ✅ **Real-time progress** - See detection as it processes
    - ✅ **Export results** - Download video and CSV reports
    """)
