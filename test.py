import cv2

# Load your test frame
img = cv2.imread("round_frame.png")
if img is None:
    raise RuntimeError("Couldn't read test.png")

h, w, _ = img.shape

# 1) Define vertical bounds for all three ROIs (same top band)
y1 = 20
y2 = 80  # height ~60px; adjust as needed

# 2) Timer ROI (centered horizontally, ~20% of width)
timer_w = int(w * 0.05)
cx = w // 2
tx1 = cx - timer_w // 2
tx2 = cx + timer_w // 2

# 3) Round‐score ROIs (same vertical bounds, each ~10% of width)
round_w = int(w * 0.03)
gap     = int(w * 0.03)  # pixels between timer and scores

# Left score just left of timer
lrx2 = tx1 - gap
lrx1 = lrx2 - round_w
# Right score just right of timer
rrx1 = tx2 + gap
rrx2 = rrx1 + round_w

# Bundle them
rois = {
    "timer":    (y1, y2, tx1, tx2),
    "round_L":  (y1, y2, lrx1, lrx2),
    "round_R":  (y1, y2, rrx1, rrx2),
}

# 4) Draw the boxes on a copy for visualization
vis = img.copy()
for name, (yy1, yy2, xx1, xx2) in rois.items():
    cv2.rectangle(vis, (xx1, yy1), (xx2, yy2), (0,255,0), 2)
    cv2.putText(vis, name, (xx1, yy1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# Show the full image with ROIs
cv2.imshow("ROIs", vis)

cv2.waitKey(0)
cv2.destroyAllWindows()
