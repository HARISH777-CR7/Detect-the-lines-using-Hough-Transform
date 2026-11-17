#  Lane Detection

##  Aim

To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

---

## Learning Objective

* Understand each stage of image processing
* Learn how to build a complete computer vision pipeline
* Practice writing code in guided sections

**Important Instruction:**
👉 Write code **ONLY in places marked as `# Your Code Here`**
👉 Do NOT modify any other part of the code

---
##  Instructions

* Fill ONLY in `# Your Code Here` sections
* Do NOT change existing code
* Run step-by-step
* Verify outputs

##  Software Used

* Anaconda – Python 3.7
* Jupyter Notebook / VS Code
* OpenCV (cv2)
* NumPy
* Matplotlib

---

##  Developed By

* **Name:** HARISH K
* **Register No:** 212225230094

##  Algorithm & Explanation

---

###  Step 1: Import Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

###  Step 2: Read the Image

```python
img = cv2.imread("Downloads/WAY.jpg")
```

---

###  Step 3: Convert to Grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

###  Step 4: Display Images

```python
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
```

---

###  Step 5: Thresholding

```python
threshold = int(np.percentile(gray, 90))
print("Threshold used:", threshold)
_, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6, 5))
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()
```

---

###  Step 6: Region of Interest (ROI)

```python
height, width = thresh.shape

# Trapezoid covering the road ahead
polygon = np.array([[
    (0, height),
    (width, height),
    (int(0.55 * width), int(0.6 * height)),
    (int(0.45 * width), int(0.6 * height))
]], dtype=np.int32)

mask = np.zeros_like(thresh)
cv2.fillPoly(mask, polygon, 255)
masked = cv2.bitwise_and(thresh, mask)

plt.figure(figsize=(6, 5))
plt.imshow(masked, cmap="gray")
plt.title("ROI Masked Image")
plt.axis("off")
plt.show()
```

---

### Step 7: Edge Detection (Canny)

```python
edges = cv2.Canny(masked, 50, 150)

plt.figure(figsize=(6, 5))
plt.imshow(edges, cmap="gray")
plt.title("Edge Detected Image")
plt.axis("off")
plt.show()
```

---

###  Step 8: Gaussian Blur

```python
blur = cv2.GaussianBlur(edges, (5, 5), 0)

plt.figure(figsize=(6, 5))
plt.imshow(blur, cmap="gray")
plt.title("Smoothed Image")
plt.axis("off")
plt.show()
```

---

###  Step 9: Hough Transform

```python
lines = cv2.HoughLinesP(blur, 1, np.pi / 180, threshold=50,
                        minLineLength=50, maxLineGap=150)

line_img = np.zeros_like(img)
if lines is not None:
    lines = lines.reshape(-1, 4)
    for x1, y1, x2, y2 in lines:
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines")
plt.axis("off")
plt.show()
```

---

### Step 10: Lane Detection Logic

```python
left_fit = []
right_fit = []

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        # Image y-axis points down: left lane has negative slope, right lane positive
        if slope < -0.3:
            left_fit.append((slope, intercept))
        elif slope > 0.3:
            right_fit.append((slope, intercept))

def make_points(y_bottom, y_top, fit):
    slope, intercept = fit
    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return (x_bottom, y_bottom, x_top, y_top)

h = img.shape[0]
y_bottom = h
y_top = int(h * 0.6)

lane_img = np.zeros_like(img)

if len(left_fit) > 0:
    x1, y1, x2, y2 = make_points(y_bottom, y_top, np.mean(left_fit, axis=0))
    cv2.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 8)

if len(right_fit) > 0:
    x1, y1, x2, y2 = make_points(y_bottom, y_top, np.mean(right_fit, axis=0))
    cv2.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 8)

final = cv2.addWeighted(img, 0.8, lane_img, 1, 0)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.title("Final Lane Detection Output")
plt.axis("off")
plt.show()
```

---

##  Expected Output

* Original image

<img width="340" height="225" alt="image" src="https://github.com/user-attachments/assets/6e9fd9a4-b3ed-4b49-a40e-34820366777f" />

* Grayscale image

<img width="353" height="226" alt="image" src="https://github.com/user-attachments/assets/566893ca-fe30-4831-b6a7-683d062cce96" />

* Thresholded image

<img width="647" height="428" alt="image" src="https://github.com/user-attachments/assets/ed9f134d-858e-443a-b9a6-0aed2044d978" />

* ROI masked image

<img width="633" height="402" alt="image" src="https://github.com/user-attachments/assets/b928dc09-5b6a-4057-b4c8-64cd8e218a36" />

* Edge detected image

<img width="637" height="407" alt="image" src="https://github.com/user-attachments/assets/51f77804-6af2-464f-9873-dca5acacf274" />

* Smoothed image

<img width="620" height="407" alt="image" src="https://github.com/user-attachments/assets/2c7567fe-fcbb-4c69-88d9-d04514a1c53b" />

* Detected lines

<img width="626" height="407" alt="image" src="https://github.com/user-attachments/assets/e5011756-a936-488b-a515-07c47bd531aa" />

* Final lane detection output

<img width="810" height="525" alt="image" src="https://github.com/user-attachments/assets/99d9d93b-5184-4f75-9261-36cfceba05c9" />


---



---

## Result

Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

---


