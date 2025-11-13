import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import tempfile
import os
from datetime import datetime
import io

class TimeBasedVehicleCounter:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize time-based vehicle counter"""
        self.model = YOLO(model_path)
        
        self.vehicle_classes = {
            2: 'Car',
            3: 'Motorcycle', 
            5: 'Bus',
            7: 'Truck'
        }
        
        self.time_counts = {}
        self.counted_ids = set()
        self.vehicle_tracks = {}
        self.counting_line = None
        self.line_margin = 30
        
    def setup_counting_line(self, frame_width, frame_height, position=0.5, orientation='horizontal'):
        """Setup single counting line"""
        if orientation == 'horizontal':
            self.counting_line = {
                'type': 'horizontal',
                'y': int(frame_height * position),
                'x1': 0,
                'x2': frame_width,
                'color': (0, 255, 0)
            }
        else:
            self.counting_line = {
                'type': 'vertical',
                'x': int(frame_width * position),
                'y1': 0,
                'y2': frame_height,
                'color': (0, 255, 0)
            }
    
    def get_time_slot(self, frame_number, fps):
        """Get 30-minute time slot for a frame"""
        seconds = frame_number / fps
        minutes = int(seconds / 60)
        slot_minutes = (minutes // 30) * 30
        hours = slot_minutes // 60
        mins = slot_minutes % 60
        period = "AM" if hours < 12 else "PM"
        display_hour = hours % 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:{mins:02d} {period}"
    
    def get_center_point(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    def has_crossed_line(self, track_id, current_center):
        """Check if vehicle crossed the counting line"""
        if track_id not in self.vehicle_tracks:
            return False
        
        positions = self.vehicle_tracks[track_id]['positions']
        if len(positions) < 2:
            return False
        
        prev_x, prev_y = positions[-2]
        curr_x, curr_y = current_center
        
        if self.counting_line['type'] == 'horizontal':
            line_y = self.counting_line['y']
            if ((prev_y < line_y - self.line_margin and curr_y > line_y + self.line_margin) or
                (prev_y > line_y + self.line_margin and curr_y < line_y - self.line_margin)):
                return True
        else:
            line_x = self.counting_line['x']
            if ((prev_x < line_x - self.line_margin and curr_x > line_x + self.line_margin) or
                (prev_x > line_x + self.line_margin and curr_x < line_x - self.line_margin)):
                return True
        
        return False
    
    def classify_vehicle_fhwa(self, bbox, class_id):
        """Classify vehicle into FHWA 13 classes"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        area = width * height
        
        if class_id == 3:
            return 'Class 1'
        elif class_id == 5:
            return 'Class 4'
        elif class_id == 7:
            if aspect_ratio > 2.5:
                if area > 25000:
                    return 'Class 13'
                elif area > 20000:
                    return 'Class 12'
                else:
                    return 'Class 11'
            elif aspect_ratio > 2.0:
                if area > 22000:
                    return 'Class 10'
                elif area > 18000:
                    return 'Class 9'
                else:
                    return 'Class 8'
            elif aspect_ratio > 1.5:
                if area > 16000:
                    return 'Class 7'
                elif area > 12000:
                    return 'Class 6'
                else:
                    return 'Class 5'
            else:
                return 'Class 5'
        elif class_id == 2:
            if aspect_ratio > 1.7 and area > 13000:
                return 'Class 3'
            else:
                return 'Class 2'
        
        return 'Class 2'
    
    def get_class_description(self, class_num):
        """Get description for FHWA class"""
        descriptions = {
            'Class 1': 'Motorcycle',
            'Class 2': 'Passenger Car',
            'Class 3': '2-Axle 4 Tire Single Unit',
            'Class 4': 'Bus',
            'Class 5': '2-Axle 6 Tire Single Unit',
            'Class 6': '3-Axle Single Unit',
            'Class 7': '4+ Axle Single Unit',
            'Class 8': '4 Axle Single Trailer',
            'Class 9': '5 Axle Single Trailer',
            'Class 10': '6+ Axle Single Trailer',
            'Class 11': '<5 Axle Multi Trailer',
            'Class 12': '6 Axle Multi Trailer',
            'Class 13': '7+ Axle Multi Trailer'
        }
        return descriptions.get(class_num, 'Unknown')
    
    def process_video_streamlit(self, video_path, confidence_threshold=0.3, 
                                 skip_frames=2, line_position=0.5, 
                                 line_orientation='horizontal', progress_bar=None):
        """Process video for Streamlit with progress updates"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.setup_counting_line(frame_width, frame_height, line_position, line_orientation)
        
        frame_count = 0
        max_history = 30
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            if progress_bar:
                progress_bar.progress(frame_count / total_frames)
            
            if frame_count % skip_frames != 0:
                continue
            
            current_time_slot = self.get_time_slot(frame_count, fps)
            
            if current_time_slot not in self.time_counts:
                self.time_counts[current_time_slot] = {
                    f'Class {i}': 0 for i in range(1, 14)
                }
            
            results = self.model.track(
                frame,
                persist=True,
                conf=confidence_threshold,
                classes=[2, 3, 5, 7],
                verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, cls, track_id in zip(boxes, classes, track_ids):
                    cls_id = int(cls)
                    center = self.get_center_point(box)
                    
                    if track_id not in self.vehicle_tracks:
                        self.vehicle_tracks[track_id] = {
                            'positions': [],
                            'class': self.classify_vehicle_fhwa(box, cls_id),
                            'counted': False,
                            'last_seen': frame_count
                        }
                    
                    self.vehicle_tracks[track_id]['positions'].append(center)
                    self.vehicle_tracks[track_id]['last_seen'] = frame_count
                    
                    if len(self.vehicle_tracks[track_id]['positions']) > max_history:
                        self.vehicle_tracks[track_id]['positions'].pop(0)
                    
                    if not self.vehicle_tracks[track_id]['counted']:
                        if self.has_crossed_line(track_id, center):
                            vehicle_class = self.vehicle_tracks[track_id]['class']
                            self.time_counts[current_time_slot][vehicle_class] += 1
                            self.counted_ids.add(track_id)
                            self.vehicle_tracks[track_id]['counted'] = True
        
        cap.release()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.time_counts).T
        df['Total'] = df.sum(axis=1)
        df.index.name = 'Time Slot'
        
        return df

# Streamlit App
def main():
    st.set_page_config(
        page_title="Vehicle Counter",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 AI Vehicle Counter - FHWA Classification")
    st.markdown("### Upload a traffic video and get automated vehicle counts by 30-minute intervals")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.3, 0.05)
        skip_frames = st.slider("Skip Frames (Speed)", 1, 10, 2)
        line_position = st.slider("Counting Line Position", 0.0, 1.0, 0.5, 0.05)
        line_orientation = st.selectbox("Line Orientation", ["horizontal", "vertical"])
        
        st.markdown("---")
        st.markdown("### 📊 FHWA Vehicle Classes")
        st.markdown("""
        - **Class 1**: Motorcycle
        - **Class 2**: Passenger Car
        - **Class 3**: Pickup/Van
        - **Class 4**: Bus
        - **Class 5-13**: Trucks (by size)
        """)
    
    # Main content
    uploaded_file = st.file_uploader("📤 Upload Traffic Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("▶️ Start Processing", type="primary"):
            with st.spinner("🔄 Processing video... This may take a few minutes"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    counter = TimeBasedVehicleCounter()
                    
                    df = counter.process_video_streamlit(
                        video_path,
                        confidence_threshold=confidence,
                        skip_frames=skip_frames,
                        line_position=line_position,
                        line_orientation=line_orientation,
                        progress_bar=progress_bar
                    )
                    
                    progress_bar.empty()
                    st.success(f"✅ Processing Complete! Total vehicles counted: {len(counter.counted_ids)}")
                    
                    # Display results
                    st.header("📊 Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Vehicles", len(counter.counted_ids))
                    with col2:
                        st.metric("Time Slots", len(df))
                    with col3:
                        st.metric("Peak Count", df['Total'].max())
                    
                    st.subheader("📈 Counts by Time Slot")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download CSV
                    csv = df.to_csv()
                    st.download_button(
                        label="💾 Download CSV Report",
                        data=csv,
                        file_name=f"vehicle_counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    st.subheader("📊 Visualization")
                    st.line_chart(df['Total'])
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    os.unlink(video_path)
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit & YOLOv8")

if __name__ == "__main__":
    main()
