
# ⚽ Football Match Object Detection

![Final Output Screenshot](https://github.com/jamesbcn/football-object-detection/blob/main/screenshot.png)

## Overview

This project uses computer vision and machine learning to analyze football match footage. Leveraging **YOLOv5** for object detection, it tracks players and the ball, assigns players to teams using colour-based clustering, estimates camera movement, and performs a perspective transformation to calculate player speed and distances covered throughout the match.

This project was completed as part of an object detection course.

---

## 🔍 Features

- **Player and Ball Detection**: Using YOLOv5 to detect all players and the ball in each frame.
- **Tracking Across Frames**: Assigns consistent IDs to players and the ball to maintain tracking over time.
- **Team Classification**: K-means clustering on jersey colours to separate teams.
- **Ball Interpolation**: Estimates ball location when temporarily undetected.
- **Camera Motion Estimation**: Uses optical flow to stabilize analysis in the presence of camera panning or zoom.
- **Perspective Transformation**: Maps the video to a top-down pitch view using homography.
- **Speed & Distance Calculation**: Converts pixel motion into real-world player speed and total distance covered.

---

## 📹 Sample Input & Output

- 🔗 [Sample Input Video](https://drive.google.com/file/d/1uf2J619om7qdJTWUVufbi9J7l7loluus/view?usp=sharing)  
- 🎥 [Final Processed Output Video](https://drive.google.com/file/d/1Phz1a9tf0buqNZhtOXROZeccWNmKoY9R/view?usp=sharing)

---

## 🛠️ Tech Stack

- **Language**: Python
- **Computer Vision**: YOLOv5, OpenCV
- **Data Analysis**: NumPy, Matplotlib
- **Clustering**: K-means (scikit-learn)
- **Development**: Jupyter Notebook, Visual Studio Code

---

## 🚀 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jamesbcn/football-object-detection.git
   cd football-object-detection
   ```

2. Create and activate a virtual environment (optional)

   ```bash
   python -m venv venv
   source venv/bin/activate      # On macOS/Linux
   venv\Scripts\activate         # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the analysis:

   ```bash
   python main.py
   ```


---

## 🌟 Future Improvements

1. **Goalkeeper Detection Enhancement**  
   Improve robustness when the goalkeeper's jersey colour closely matches the opposing team’s outfield players.

2. **Cutaway Scene Handling**  
   Add scene change detection to ignore replays, crowd shots, or broadcast overlays that disrupt pitch-based tracking.

3. **Adaptive Perspective Transformation**  
   Dynamically select and update pitch reference points (e.g., corner flags, pitch lines) depending on camera angle and visibility, improving player speed accuracy and increasing valid frame coverage.

---

## 🧠 What I Learned

- Advanced object tracking using YOLOv5
- Real-world speed estimation from pixel-level data
- Handling noisy data like camera movements and occlusions
- Applying clustering for visual feature grouping
