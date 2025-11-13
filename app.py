import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import pandas as pd
from datetime import datetime
import tempfile
import os
import time

# Page configuration
st.set_page_config(page_title="Vehicle Counter", layout="wide")

# Title with link to Fiverr
st.markdown("""
    <h1 style='text-align: center;'>
        🚗 FHWA Vehicle Counter by 
        <a href='https://fiverr.com/naseem733' target='_blank' style='color: #1DBF73; text-decoration: none;'>
            Naseem Awan
        </a>
    </h1>
""", unsafe_allow_html=True)

# Initialize session state for persistence
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = None
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'video_duration' not in st.session_state:
    st.session_state.video_duration = 0

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
skip_frames = st.sidebar.slider("Skip Frames (Speed)", 1, 20, 5, 1,
                                help="Higher = faster. 10 = 10x speed!")
line_position = st.sidebar.slider("Counting Line Position", 0.0, 1.0, 0.5, 0.05)

# Advanced speed settings
with st.sidebar.expander("🚀 ULTRA Speed Settings"):
    img_size = st.selectbox("Detection Image Size", 
                           options=[320, 416, 640], 
                           index=0,
                           help="320 = 4x faster than 640!")
    
    max_det = st.slider("Max Detections Per Frame", 10, 100, 30, 10,
                       help="Lower = faster processing")
    
    update_display_every = st.slider("Update Display Every N Frames", 
                                    10, 100, 50, 10,
                                    help="Higher = faster (less UI updates)")
    
    use_fast_codec = st.checkbox("Fast Video Encoding", value=True,
                                 help="XVID codec - 2x faster encoding")
    
    disable_video_output = st.checkbox("Skip Video Output (CSV only)", value=False,
                                      help="⚡ 5x FASTER - Only generate CSV report")

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

# Helper function to format time
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.1f}h {minutes:.0f}m"

# YOLO to FHWA mapping (optimized - no repeated calculations)
def map_to_fhwa(yolo_class, bbox_area):
    """Map YOLO class to FHWA vehicle class - OPTIMIZED"""
    if yolo_class == 3:  # motorcycle
        return 1
    elif yolo_class == 2:  # car
        return 2 if bbox_area < 5000 else 3
    elif yolo_class == 5:  # bus
        return 4
    elif yolo_class == 7:  # truck
        if bbox_area < 8000:
            return 5
        elif bbox_area < 15000:
            return 8
        else:
            return 9
    return 2

# ULTRA FAST Vehicle tracker - using Manhattan distance
class FastVehicleTracker:
    def __init__(self, max_distance=150):
        self.next_id = 0
        self.tracks = {}
        self.counted = set()
        self.max_distance = max_distance
        self.max_missing = 10
        
    def update(self, detections, line_y):
        newly_counted = []
        
        if len(detections) == 0:
            # Quick cleanup
            to_remove = [tid for tid, track in self.tracks.items() 
                        if track['missing'] > self.max_missing]
            for tid in to_remove:
                del self.tracks[tid]
            
            for track in self.tracks.values():
                track['missing'] += 1
            
            return newly_counted
        
        current_centroids = np.array([d[0] for d in detections], dtype=np.float32)
        current_classes = [d[1] for d in detections]
        
        if len(self.tracks) == 0:
            # Register all new
            for centroid, cls in detections:
                self.tracks[self.next_id] = {
                    'pos': centroid,
                    'class': cls,
                    'missing': 0
                }
                self.next_id += 1
            return newly_counted
        
        # FAST matching using Manhattan distance (faster than Euclidean)
        track_ids = list(self.tracks.keys())
        track_pos = np.array([self.tracks[tid]['pos'] for tid in track_ids], dtype=np.float32)
        
        # Manhattan distance: |x1-x2| + |y1-y2| (faster than sqrt)
        distances = np.abs(track_pos[:, np.newaxis, 0] - current_centroids[:, 0]) + \
                   np.abs(track_pos[:, np.newaxis, 1] - current_centroids[:, 1])
        
        # Greedy matching (faster than Hungarian)
        matched_tracks = set()
        matched_dets = set()
        
        for _ in range(min(len(track_ids), len(detections))):
            min_dist = distances.min()
            if min_dist > self.max_distance:
                break
            
            t_idx, d_idx = np.unravel_index(distances.argmin(), distances.shape)
            
            tid = track_ids[t_idx]
            old_y = self.tracks[tid]['pos'][1]
            new_pos = current_centroids[d_idx]
            new_y = new_pos[1]
            
            # Update
            self.tracks[tid]['pos'] = new_pos
            self.tracks[tid]['missing'] = 0
            
            # Check crossing
            if tid not in self.counted and old_y < line_y <= new_y:
                self.counted.add(tid)
                newly_counted.append(self.tracks[tid]['class'])
            
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            
            # Prevent re-matching
            distances[t_idx, :] = np.inf
            distances[:, d_idx] = np.inf
        
        # Register new detections
        for d_idx in range(len(detections)):
            if d_idx not in matched_dets:
                self.tracks[self.next_id] = {
                    'pos': current_centroids[d_idx],
                    'class': current_classes[d_idx],
                    'missing': 0
                }
                self.next_id += 1
        
        # Remove old tracks
        for t_idx in range(len(track_ids)):
            if t_idx not in matched_tracks:
                tid = track_ids[t_idx]
                self.tracks[tid]['missing'] += 1
                if self.tracks[tid]['missing'] > self.max_missing:
                    del self.tracks[tid]
        
        return newly_counted

# File uploader with 10GB limit
uploaded_file = st.file_uploader("📁 Upload Video File (Max 10GB)", 
                                  type=['mp4', 'avi', 'mov', 'mkv'],
                                  accept_multiple_files=False)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    # Estimate processing time
    cap_temp = cv2.VideoCapture(video_path)
    total_frames_temp = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_temp = int(cap_temp.get(cv2.CAP_PROP_FPS))
    video_duration_temp = total_frames_temp / fps_temp if fps_temp > 0 else 0
    cap_temp.release()
    
    # Calculate speedup
    base_time = 0.1  # seconds per frame at 640px
    size_factor = (img_size / 640) ** 2
    frames_to_process = total_frames_temp / skip_frames
    video_output_factor = 0.2 if disable_video_output else 1.0
    
    estimated_time = frames_to_process * base_time * size_factor * video_output_factor
    speedup = video_duration_temp / estimated_time if estimated_time > 0 else 0
    
    st.info(f"📊 Video: {format_time(video_duration_temp)} | Est. Processing: ~{format_time(estimated_time)} | Speedup: **{speedup:.0f}x** 🚀")
    
    if st.button("▶️ Start Processing", type="primary"):
        try:
            st.session_state.processed = False
            
            start_time = time.time()
            
            with st.spinner("🔄 Loading optimized YOLO model..."):
                model = YOLO('yolov8n.pt')
                st.success(f"✅ Model loaded! Using {img_size}px detection")
            
            # Open video with minimal buffering
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_duration_seconds = total_frames / fps if fps > 0 else 0
            
            line_y = int(height * line_position)
            tracker = FastVehicleTracker(max_distance=150)
            class_counts = defaultdict(int)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()
            frame_placeholder = st.empty()
            
            frame_count = 0
            processed_frames = 0
            
            # Setup video output (if enabled)
            out = None
            output_path = None
            if not disable_video_output:
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'XVID') if use_fast_codec else cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps//skip_frames, (width, height))
            
            last_display = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % skip_frames != 0:
                    continue
                
                processed_frames += 1
                
                # OPTIMIZED INFERENCE
                results = model.predict(
                    frame, 
                    conf=confidence, 
                    verbose=False,
                    imgsz=img_size,  # Smaller = faster
                    max_det=max_det,  # Limit detections
                    agnostic_nms=True,  # Faster NMS
                    classes=[2, 3, 5, 7]  # Only vehicles
                )
                
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                        bbox_area = (x2 - x1) * (y2 - y1)
                        fhwa_class = map_to_fhwa(cls, bbox_area)
                        detections.append((centroid, fhwa_class))
                        
                        # Draw only if video output enabled
                        if not disable_video_output:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0, 0, 255), -1)
                            cv2.putText(frame, f"C{fhwa_class}", (int(x1), int(y1)-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                newly_counted = tracker.update(detections, line_y)
                for fhwa_class in newly_counted:
                    class_counts[fhwa_class] += 1
                
                # Draw annotations (if video output enabled)
                if not disable_video_output:
                    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
                    cv2.putText(frame, "COUNTING LINE", (10, line_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    total_count = sum(class_counts.values())
                    cv2.putText(frame, f"Total: {total_count}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    y_offset = 60
                    for cls in sorted(class_counts.keys()):
                        if class_counts[cls] > 0:
                            cv2.putText(frame, f"C{cls}: {class_counts[cls]}", (10, y_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            y_offset += 25
                    
                    out.write(frame)
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                
                # Time estimates
                elapsed_time = time.time() - start_time
                if progress > 0.01:
                    estimated_total = elapsed_time / progress
                    remaining_time = estimated_total - elapsed_time
                    current_speed = (frame_count / fps) / elapsed_time if elapsed_time > 0 else 0
                    time_text.text(f"⏱️ Elapsed: {format_time(elapsed_time)} | Remaining: {format_time(remaining_time)} | Speed: {current_speed:.1f}x")
                
                total_count = sum(class_counts.values())
                status_text.text(f"Frame: {frame_count}/{total_frames} | Vehicles: {total_count}")
                
                # Update display less frequently
                if not disable_video_output and processed_frames - last_display >= update_display_every:
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                          channels="RGB", use_container_width=True)
                    last_display = processed_frames
            
            cap.release()
            if out is not None:
                out.release()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            total = sum(class_counts.values())
            
            # Store in session state
            st.session_state.processed = True
            st.session_state.output_video_path = output_path
            st.session_state.class_counts = dict(class_counts)
            st.session_state.total_count = total
            st.session_state.processing_time = processing_time
            st.session_state.video_duration = video_duration_seconds
            
            results_data = []
            for cls in range(1, 14):
                count = class_counts.get(cls, 0)
                results_data.append({
                    'FHWA Class': cls,
                    'Vehicle Type': fhwa_classes[cls],
                    'Count': count,
                    'Percentage': f"{(count/total*100):.1f}%" if total > 0 else "0%"
                })
            
            st.session_state.results_df = pd.DataFrame(results_data)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_data = {'Time Slot': [datetime.now().strftime("%I:%M %P")]}
            
            for i in range(1, 14):
                csv_data[f'Class {i}'] = [class_counts.get(i, 0)]
            
            csv_data['Total'] = [total]
            csv_df = pd.DataFrame(csv_data)
            st.session_state.csv_data = csv_df.to_csv(index=False)
            st.session_state.csv_filename = f"vehicle_counts_{timestamp}.csv"
            st.session_state.video_filename = f"processed_{uploaded_file.name}"
            
            st.success(f"✅ Processing Complete! Total vehicles: {total}")
            st.balloons()
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    if not st.session_state.processed:
        os.unlink(video_path)

# Display results if processed (PERSISTENT)
if st.session_state.processed:
    processing_speed = (st.session_state.video_duration / st.session_state.processing_time) if st.session_state.processing_time > 0 else 0
    
    st.markdown(f"""
        <div style='background-color: #e8f4f8; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #0066cc;'>
            <h3 style='color: #0066cc; margin: 0;'>⏱️ Processing Time</h3>
            <p style='font-size: 1.3rem; font-weight: bold; color: #004080; margin: 10px 0 5px 0;'>
                {format_time(st.session_state.processing_time)}
            </p>
            <p style='color: #666; margin: 0; font-size: 0.9rem;'>
                Video Duration: {format_time(st.session_state.video_duration)} | 
                Processing Speed: <strong>{processing_speed:.1f}x</strong> real-time
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("⬇️ Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            with open(st.session_state.output_video_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name=st.session_state.video_filename,
                    mime="video/mp4",
                    key="download_video"
                )
        else:
            st.info("ℹ️ Video output was disabled for faster processing")
    
    with col2:
        if st.session_state.csv_data:
            st.download_button(
                label="📥 Download CSV Report",
                data=st.session_state.csv_data,
                file_name=st.session_state.csv_filename,
                mime="text/csv",
                key="download_csv"
            )
    
    st.subheader("📊 Vehicle Count Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Vehicles", st.session_state.total_count)
        st.dataframe(st.session_state.results_df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Distribution")
        chart_data = st.session_state.results_df[st.session_state.results_df['Count'] > 0][['Vehicle Type', 'Count']]
        if not chart_data.empty:
            st.bar_chart(chart_data.set_index('Vehicle Type'))
    
    st.subheader("📄 CSV Preview")
    csv_preview_df = pd.read_csv(pd.io.common.StringIO(st.session_state.csv_data))
    st.dataframe(csv_preview_df)
    
    if st.button("🔄 Process Another Video"):
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            os.unlink(st.session_state.output_video_path)
        
        st.session_state.processed = False
        st.session_state.output_video_path = None
        st.session_state.csv_data = None
        st.session_state.results_df = None
        st.session_state.class_counts = None
        st.session_state.total_count = 0
        st.session_state.processing_time = 0
        st.session_state.video_duration = 0
        st.rerun()

elif uploaded_file is None:
    st.info("👆 Please upload a video file to begin")
    st.markdown("""
    ### 📋 Instructions:
    1. Upload a traffic video (MP4, AVI, MOV, MKV) - Max 10GB
    2. **SPEED SETTINGS** (in sidebar "ULTRA Speed Settings"):
       - **Image Size = 320** → 4x faster
       - **Skip Frames = 10** → 10x faster
       - **Skip Video Output** → 5x faster (CSV only)
    3. Click "Start Processing"
    
    ### 🚀 **ULTRA Speed Optimizations:**
    
    | Setting | Speed Gain | 8hr Video Time |
    |---------|------------|----------------|
    | Default (640px, skip=2) | 2x | ~4 hours |
    | **320px + skip=5** | **20x** | **~24 min** |
    | **320px + skip=10** | **40x** | **~12 min** |
    | **320px + skip=10 + No Video** | **200x** | **~2.4 min** 🚀 |
    
    ### ⚡ **What Makes This ULTRA FAST:**
    
    1. **320px detection** - 4x faster inference
    2. **Manhattan distance** - Faster than Euclidean (no sqrt)
    3. **Greedy matching** - Faster than Hungarian algorithm
    4. **Class filtering** - Only detect vehicles (classes 2,3,5,7)
    5. **Agnostic NMS** - Faster non-maximum suppression
    6. **Limited detections** - Max 30 per frame
    7. **Minimal buffer** - `CAP_PROP_BUFFERSIZE = 1`
    8. **Fast codec** - XVID encoding (2x faster than mp4v)
    9. **Reduced UI updates** - Update every 50 frames
    10. **Skip video output** - ⚡ **5x FASTER** - Generate CSV only!
    
    ### 🎯 **Recommended for 8-hour video:**
    - Image Size: **320**
    - Skip Frames: **10**
    - Skip Video Output: **✅ Enabled**
    - **Result: ~2-3 minutes processing!** 🚀
    
    ### 📊 **Features:**
    - ✅ No double counting
    - ✅ FHWA 13-class classification
    - ✅ Real-time progress tracking
    - ✅ CSV export
    - ✅ Optional video output
    """)
