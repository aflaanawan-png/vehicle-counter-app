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
from multiprocessing import Process, Queue, Manager
import subprocess
import yt_dlp

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
skip_frames = st.sidebar.slider("Skip Frames (Speed)", 1, 10, 2)

# Parallel processing option
st.sidebar.header("🚀 Processing Mode")
processing_mode = st.sidebar.radio(
    "Select Processing Mode",
    options=["Single Thread", "Parallel (2x Speed)"],
    index=0,
    help="Parallel mode splits video into 2 parts and processes simultaneously for ~2x speedup"
)
enable_parallel = (processing_mode == "Parallel (2x Speed)")

if enable_parallel:
    st.sidebar.success("⚡ Parallel mode enabled - ~2x faster!")
else:
    st.sidebar.info("🔄 Single thread mode - Standard processing")

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

# YOLO to FHWA mapping (IMPROVED for large vehicles)
def map_to_fhwa(yolo_class, bbox_area, bbox_width, bbox_height):
    """Map YOLO class to FHWA vehicle class with improved detection"""
    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1
    
    if yolo_class == 3:  # motorcycle
        return 1
    elif yolo_class == 2:  # car
        # Larger cars = pickup/van
        if bbox_area > 6000 or aspect_ratio > 2.0:
            return 3  # Pickup/Van
        return 2  # Passenger Car
    elif yolo_class == 5:  # bus
        return 4
    elif yolo_class == 7:  # truck
        # Improved truck classification based on size
        if bbox_area < 10000:
            return 5  # Single Unit Truck (2-axle)
        elif bbox_area < 18000:
            return 8  # Single Trailer Truck (3-4 axle)
        elif bbox_area < 28000:
            return 9  # Single Trailer Truck (5-axle)
        else:
            return 10  # Single Trailer Truck (6+ axle)
    return 2

# Vehicle tracker (NO COUNTING LINE - counts when first detected)
class VehicleTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.counted = set()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, fhwa_class):
        """Register new vehicle and count it immediately"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'class': fhwa_class
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1  # Return the ID of newly registered vehicle
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections):
        """Update tracker and return newly counted vehicles"""
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
            # First detections - register and count all
            for i, (centroid, fhwa_class) in enumerate(detections):
                new_id = self.register(centroid, fhwa_class)
                if new_id not in self.counted:
                    self.counted.add(new_id)
                    newly_counted.append(fhwa_class)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid]['centroid'] for oid in object_ids])
            
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            
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
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # New detections - register and count
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                new_id = self.register(input_centroids[col], input_classes[col])
                if new_id not in self.counted:
                    self.counted.add(new_id)
                    newly_counted.append(input_classes[col])
        
        return newly_counted

# Function to download YouTube video
def download_youtube_video(url, progress_callback=None):
    """Download YouTube video and return local path"""
    try:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [progress_callback] if progress_callback else [],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        return output_path, None
    except Exception as e:
        return None, str(e)

# Function to split video into two parts using ffmpeg
def split_video(input_path, output_path1, output_path2):
    """Split video into two equal parts using ffmpeg"""
    try:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        mid_point = duration / 2
        
        cmd1 = [
            'ffmpeg', '-i', input_path,
            '-t', str(mid_point),
            '-c', 'copy',
            '-y', output_path1
        ]
        
        cmd2 = [
            'ffmpeg', '-i', input_path,
            '-ss', str(mid_point),
            '-c', 'copy',
            '-y', output_path2
        ]
        
        subprocess.run(cmd1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        return True, duration
    except Exception as e:
        st.error(f"Error splitting video: {e}")
        return False, 0

# Process video segment (for parallel processing)
def process_video_segment(video_path, confidence, skip_frames, 
                         segment_id, result_queue, progress_dict):
    """Process a video segment and return results"""
    try:
        model = YOLO('yolov8n.pt')
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        tracker = VehicleTracker(max_disappeared=fps, max_distance=150)
        class_counts = defaultdict(int)
        
        frame_count = 0
        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps//skip_frames, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % skip_frames != 0:
                continue
            
            results = model(frame, conf=confidence, verbose=False)
            
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in [2, 3, 5, 7]:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                        bbox_area = (x2 - x1) * (y2 - y1)
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        fhwa_class = map_to_fhwa(cls, bbox_area, bbox_width, bbox_height)
                        detections.append((centroid, fhwa_class))
                        
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"Class {fhwa_class}", (int(x1), int(y1)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            newly_counted = tracker.update(detections)
            for fhwa_class in newly_counted:
                class_counts[fhwa_class] += 1
            
            cv2.putText(frame, f"SEGMENT {segment_id}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            total_count = sum(class_counts.values())
            cv2.putText(frame, f"Total: {total_count}", (10, 65),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            y_offset = 100
            for cls, count in sorted(class_counts.items()):
                if count > 0:
                    cv2.putText(frame, f"C{cls}: {count}", (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
            
            out.write(frame)
            
            progress = frame_count / total_frames
            progress_dict[f'segment_{segment_id}'] = progress
        
        cap.release()
        out.release()
        
        result_queue.put({
            'segment_id': segment_id,
            'class_counts': dict(class_counts),
            'output_path': output_path
        })
        
    except Exception as e:
        result_queue.put({
            'segment_id': segment_id,
            'error': str(e)
        })

# Merge two videos using ffmpeg
def merge_videos(video1_path, video2_path, output_path):
    """Merge two videos using ffmpeg"""
    try:
        concat_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        concat_file.write(f"file '{video1_path}'\n")
        concat_file.write(f"file '{video2_path}'\n")
        concat_file.close()
        
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', concat_file.name,
            '-c', 'copy',
            '-y', output_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        os.unlink(concat_file.name)
        return True
    except Exception as e:
        st.error(f"Error merging videos: {e}")
        return False

# Video input options
st.subheader("📹 Video Input")
input_method = st.radio(
    "Select Input Method",
    options=["Upload Video File", "YouTube URL"],
    horizontal=True
)

video_path = None

if input_method == "Upload Video File":
    uploaded_file = st.file_uploader("📁 Upload Video File (Max 10GB)", 
                                      type=['mp4', 'avi', 'mov', 'mkv'],
                                      accept_multiple_files=False)
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.video(video_path)

else:  # YouTube URL
    youtube_url = st.text_input("🔗 Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    if youtube_url:
        if st.button("📥 Download Video", type="secondary"):
            with st.spinner("⬇️ Downloading video from YouTube..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_hook(d):
                    if d['status'] == 'downloading':
                        try:
                            percent = d.get('downloaded_bytes', 0) / d.get('total_bytes', 1) * 100
                            progress_bar.progress(min(percent / 100, 1.0))
                            status_text.text(f"Downloading: {percent:.1f}%")
                        except:
                            pass
                
                video_path, error = download_youtube_video(youtube_url, progress_hook)
                
                if error:
                    st.error(f"❌ Download failed: {error}")
                    video_path = None
                else:
                    progress_bar.progress(1.0)
                    st.success("✅ Download complete!")
                    st.video(video_path)

if video_path is not None:
    # Check video duration
    cap_temp = cv2.VideoCapture(video_path)
    total_frames_temp = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_temp = int(cap_temp.get(cv2.CAP_PROP_FPS))
    video_duration_temp = total_frames_temp / fps_temp if fps_temp > 0 else 0
    cap_temp.release()
    
    # Display info based on mode
    if enable_parallel:
        estimated_time = video_duration_temp / (skip_frames * 2)
        st.info(f"🚀 **Parallel Mode** | Video: {format_time(video_duration_temp)} | Est. Processing: ~{format_time(estimated_time)} | Speedup: ~2x")
    else:
        estimated_time = video_duration_temp / skip_frames
        st.info(f"📊 **Single Thread Mode** | Video: {format_time(video_duration_temp)} | Est. Processing: ~{format_time(estimated_time)}")
    
    if st.button("▶️ Start Processing", type="primary"):
        try:
            st.session_state.processed = False
            start_time = time.time()
            
            # Check if we should use parallel processing
            if enable_parallel:
                st.info("🔄 Splitting video into 2 segments for parallel processing...")
                
                segment1_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                segment2_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                
                with st.spinner("✂️ Splitting video..."):
                    success, duration = split_video(video_path, segment1_path, segment2_path)
                
                if not success:
                    st.error("Failed to split video. Falling back to single-threaded processing.")
                    enable_parallel = False
                else:
                    st.success("✅ Video split complete! Starting parallel processing...")
                    
                    manager = Manager()
                    progress_dict = manager.dict()
                    progress_dict['segment_1'] = 0.0
                    progress_dict['segment_2'] = 0.0
                    
                    result_queue = Queue()
                    
                    with st.spinner("🔄 Loading YOLO models for both segments..."):
                        p1 = Process(target=process_video_segment, 
                                    args=(segment1_path, confidence, skip_frames, 
                                          1, result_queue, progress_dict))
                        p2 = Process(target=process_video_segment, 
                                    args=(segment2_path, confidence, skip_frames, 
                                          2, result_queue, progress_dict))
                        
                        p1.start()
                        p2.start()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("🎬 **Segment 1** (First Half)")
                        progress_bar1 = st.progress(0)
                    with col2:
                        st.write("🎬 **Segment 2** (Second Half)")
                        progress_bar2 = st.progress(0)
                    
                    status_text = st.empty()
                    time_text = st.empty()
                    
                    while p1.is_alive() or p2.is_alive():
                        prog1 = progress_dict.get('segment_1', 0)
                        prog2 = progress_dict.get('segment_2', 0)
                        progress_bar1.progress(min(prog1, 1.0))
                        progress_bar2.progress(min(prog2, 1.0))
                        
                        elapsed = time.time() - start_time
                        avg_progress = (prog1 + prog2) / 2
                        if avg_progress > 0.01:
                            est_total = elapsed / avg_progress
                            remaining = est_total - elapsed
                            time_text.text(f"⏱️ Elapsed: {format_time(elapsed)} | Remaining: ~{format_time(remaining)}")
                        
                        status_text.text(f"🔄 Processing both segments... Seg1: {prog1*100:.1f}% | Seg2: {prog2*100:.1f}%")
                        time.sleep(0.5)
                    
                    p1.join()
                    p2.join()
                    
                    progress_bar1.progress(1.0)
                    progress_bar2.progress(1.0)
                    
                    results = []
                    while not result_queue.empty():
                        results.append(result_queue.get())
                    
                    if len(results) != 2:
                        raise Exception("Failed to process both segments")
                    
                    results.sort(key=lambda x: x['segment_id'])
                    
                    for result in results:
                        if 'error' in result:
                            raise Exception(f"Segment {result['segment_id']} error: {result['error']}")
                    
                    class_counts = defaultdict(int)
                    for result in results:
                        for cls, count in result['class_counts'].items():
                            class_counts[cls] += count
                    
                    st.info("🔄 Merging processed video segments...")
                    
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    merge_success = merge_videos(results[0]['output_path'], 
                                                results[1]['output_path'], 
                                                output_path)
                    
                    if not merge_success:
                        st.warning("⚠️ Video merge failed, using first segment only")
                        output_path = results[0]['output_path']
                    
                    os.unlink(segment1_path)
                    os.unlink(segment2_path)
                    if merge_success:
                        try:
                            os.unlink(results[0]['output_path'])
                            os.unlink(results[1]['output_path'])
                        except:
                            pass
                    
                    video_duration_seconds = video_duration_temp
            
            # Single-threaded processing
            if not enable_parallel:
                with st.spinner("🔄 Loading YOLOv8 model..."):
                    model = YOLO('yolov8n.pt')
                
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                video_duration_seconds = total_frames / fps if fps > 0 else 0
                
                tracker = VehicleTracker(max_disappeared=fps, max_distance=150)
                class_counts = defaultdict(int)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                frame_placeholder = st.empty()
                
                frame_count = 0
                processed_frames = 0
                
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps//skip_frames, (width, height))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    if frame_count % skip_frames != 0:
                        continue
                    
                    processed_frames += 1
                    
                    results = model(frame, conf=confidence, verbose=False)
                    
                    detections = []
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            if cls in [2, 3, 5, 7]:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                                bbox_area = (x2 - x1) * (y2 - y1)
                                bbox_width = x2 - x1
                                bbox_height = y2 - y1
                                fhwa_class = map_to_fhwa(cls, bbox_area, bbox_width, bbox_height)
                                detections.append((centroid, fhwa_class))
                                
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0, 0, 255), -1)
                                cv2.putText(frame, f"Class {fhwa_class}", (int(x1), int(y1)-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    newly_counted = tracker.update(detections)
                    for fhwa_class in newly_counted:
                        class_counts[fhwa_class] += 1
                    
                    y_offset = 30
                    total_count = sum(class_counts.values())
                    cv2.putText(frame, f"Total: {total_count}", (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    for cls, count in sorted(class_counts.items()):
                        if count > 0:
                            y_offset += 35
                            cv2.putText(frame, f"Class {cls}: {count}", (10, y_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    out.write(frame)
                    
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    
                    elapsed_time = time.time() - start_time
                    if progress > 0:
                        estimated_total = elapsed_time / progress
                        remaining_time = estimated_total - elapsed_time
                        time_text.text(f"⏱️ Elapsed: {format_time(elapsed_time)} | Estimated remaining: {format_time(remaining_time)}")
                    
                    status_text.text(f"Processing: {frame_count}/{total_frames} frames | Detected: {total_count} vehicles")
                    
                    if processed_frames % 30 == 0:
                        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                              channels="RGB", use_container_width=True)
                
                cap.release()
                out.release()
                video_duration_seconds = video_duration_temp
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            total = sum(class_counts.values())
            
            st.session_state.processed = True
            st.session_state.output_video_path = output_path
            st.session_state.class_counts = dict(class_counts)
            st.session_state.total_count = total
            st.session_state.processing_time = processing_time
            st.session_state.video_duration = video_duration_seconds
            st.session_state.parallel_used = enable_parallel
            
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
            
            if input_method == "Upload Video File":
                st.session_state.video_filename = f"processed_{uploaded_file.name}"
            else:
                st.session_state.video_filename = f"processed_youtube_{timestamp}.mp4"
            
            st.success(f"✅ Processing Complete! Total vehicles counted: {total}")
            st.balloons()
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    if not st.session_state.processed and input_method == "Upload Video File":
        os.unlink(video_path)

# Display results if processed (PERSISTENT)
if st.session_state.processed:
    processing_speed = (st.session_state.video_duration / st.session_state.processing_time) if st.session_state.processing_time > 0 else 0
    
    parallel_badge = "🚀 Parallel Mode" if st.session_state.get('parallel_used', False) else "🔄 Single Thread"
    
    st.markdown(f"""
        <div style='background-color: #e8f4f8; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #0066cc;'>
            <h3 style='color: #0066cc; margin: 0;'>⏱️ Processing Time <span style='font-size: 0.8rem; background: #0066cc; color: white; padding: 3px 8px; border-radius: 5px; margin-left: 10px;'>{parallel_badge}</span></h3>
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
        st.session_state.parallel_used = False
        st.rerun()

elif video_path is None:
    st.info("👆 Please select a video input method to begin")
    st.markdown("""
    ### 📋 Instructions:
    1. **Choose input method**:
       - **Upload Video File**: MP4, AVI, MOV, MKV (Max 10GB)
       - **YouTube URL**: Paste any YouTube video URL
    2. **Select Processing Mode** in sidebar:
       - **Single Thread**: Standard processing
       - **Parallel (2x Speed)**: Split video and process simultaneously
    3. Adjust detection confidence and skip frames
    4. Click "Start Processing" to analyze
    5. Download results and CSV report
    
    ### 🎯 **New Features:**
    - ✅ **No counting line needed** - Vehicles counted when first detected
    - ✅ **YouTube support** - Download and process YouTube videos directly
    - ✅ **Improved large vehicle detection** - Better truck/bus classification
    - ✅ **Aspect ratio analysis** - More accurate vehicle type identification
    
    ### 🚀 **Parallel Processing Mode:**
    - ✅ **~2x faster** - Processes two halves simultaneously
    - ✅ **Works with ANY video length**
    - ✅ **Single output** - Merged video + combined CSV
    - ✅ **Real-time progress** - See both segments processing
    
    ### 📊 **Detection Improvements:**
    - 🚗 Better car vs pickup/van distinction
    - 🚛 Improved truck size classification
    - 🚌 Enhanced bus detection
    - 📏 Aspect ratio-based classification
    
    ### ⚙️ **Requirements:**
    - `ffmpeg` for parallel mode
    - `yt-dlp` for YouTube downloads
    - Multi-core CPU recommended for best performance
    """)
