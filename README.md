# ADR_offroad_dataset

## Keyboard Control Guide

### 1. Exit Program
- **`ESC`** or **`q`** → Exit the program

### 2. Toggle LiDAR Projection
- **`Space`** → Show / hide LiDAR projection points

### 3. Change Step Size
- **`b`** → Set step size to **1**
- **`n`** → Set step size to **5**
- **`m`** → Set step size to **10**

### 4. Global Synchronized Navigation
- **`,`** (comma) → Move LiDAR and all cameras **backward** by the current step size  
- **`.`** (period) → Move LiDAR and all cameras **forward** by the current step size

### 5. Control Mode Selection
- **`1` ~ `6`** → Select a specific camera (Cam1–Cam6) and activate **single camera control mode**
- **`l`** → Switch to **LiDAR-only control mode**
- **`c`** → Switch to **all cameras mode** (all cameras)

### 6. Data Index Navigation
- **`a`**
  - **Single camera mode**: Move the selected camera index **backward** by the step size
  - **LiDAR mode**: Move the LiDAR index **backward** by the step size
  - **Global mode**: Move all cameras **backward** by the step size
- **`d`**
  - **Single camera mode**: Move the selected camera index **forward** by the step size
  - **LiDAR mode**: Move the LiDAR index **forward** by the step size
  - **Global mode**: Move all cameras **forward** by the step size

### 7. Segment Marking
- **`s`** → Save the current indices as **segment start (startN)**
- **`e`** → Save the current indices as **segment end (endN)**
