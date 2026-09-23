import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",  # Gives a professional LaTeX-style look
    "font.size": 10,             # Base font size
    "axes.linewidth": 1.0,       # Slightly thicker axes for print
})

plt.close('all')

# ---------------------------------------------------------------
# DOMAIN MAPPING
# ---------------------------------------------------------------
x0, x1 = 0, 500        
sy0, sy1 = -100, 100   # Screen Y (Physical Z bounds)
sz0, sz1 = 0, 200      # Screen Z (Physical Y bounds)

R = 2                  
L = 20                 
yc_scr, zc_scr = 2, 0  # Pipe center (Physical Y=0, Z=0 maps to Screen Z=0, Y=0)

cm_to_in = 1 / 2.54
max_width_cm = 16.0
target_height_cm = 12.0  # Keeps a nice landscape ratio, well under the 22cm max

fig = plt.figure(figsize=(max_width_cm * cm_to_in, target_height_cm * cm_to_in))
ax = fig.add_subplot(111, projection='3d')

# ---------------------------------------------------------------
# Pad the view limits heavily to prevent congestion
# ---------------------------------------------------------------
pad_x, pad_y, pad_z = 120, 120, 120
ax.set_xlim(x0 - pad_x, x1 + pad_x)
ax.set_ylim(sy0 - pad_y, sy1 + pad_y)
ax.set_zlim(sz0 - pad_z, sz1 + pad_z)

# Lock aspect ratio to true data proportions INCLUSIVE of padding
ax.set_box_aspect((x1 - x0 + 2*pad_x, sy1 - sy0 + 2*pad_y, sz1 - sz0 + 2*pad_z))
ax.view_init(elev=20, azim=-150)

# ---------------------------------------------------------------
# Light grid lines on the "back" walls
# ---------------------------------------------------------------
grid_kw = dict(color='0.85', lw=0.6)

# Back X=0 plane
for sy in np.linspace(sy0, sy1, 5):
    ax.plot([x1, x1], [sy, sy], [sz0, sz1], **grid_kw)
for sz in np.linspace(sz0, sz1, 5):
    ax.plot([x1, x1], [sy0, sy1], [sz, sz], **grid_kw)

# Back Physical Z=-100 plane (Screen Y = -100)
for x in np.linspace(x0, x1, 11):
    ax.plot([x, x], [sy1, sy1], [sz0, sz1], **grid_kw)
for sz in np.linspace(sz0, sz1, 5):
    ax.plot([x0, x1], [sy1, sy1], [sz, sz], **grid_kw)

# Bottom Physical Y=0 plane (Screen Z = 0)
for x in np.linspace(x0, x1, 11):
    ax.plot([x, x], [sy0, sy1], [sz0, sz0], **grid_kw)
for sy in np.linspace(sy0, sy1, 5):
    ax.plot([x0, x1], [sy, sy], [sz0, sz0], **grid_kw)

# ---------------------------------------------------------------
# Box wireframe
# ---------------------------------------------------------------
verts = np.array([
    [x0, sy0, sz0], [x1, sy0, sz0], [x1, sy1, sz0], [x0, sy1, sz0],
    [x0, sy0, sz1], [x1, sy0, sz1], [x1, sy1, sz1], [x0, sy1, sz1],
])
edges = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

for i, j in edges:
    p, q = verts[i], verts[j]
    ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color='k', lw=1.2)

# ---------------------------------------------------------------
# Pressure Outlets (Entire Faces Shaded)
# ---------------------------------------------------------------
# Define the 4 corners of the entire Left (X=0) and Right (X=500) faces
face_x0 = [[x0, sy0, sz0], [x0, sy1, sz0], [x0, sy1, sz1], [x0, sy0, sz1]]
face_x1 = [[x1, sy0, sz0], [x1, sy1, sz0], [x1, sy1, sz1], [x1, sy0, sz1]]

# Add them as highly transparent filled patches
outlets = Poly3DCollection([face_x0, face_x1], 
                           alpha=0.15,            # Very subtle so it doesn't hide the grid
                           facecolors='salmon',   # Standard warm color for outlets
                           edgecolors='none')     # No extra borders to avoid clutter
ax.add_collection3d(outlets)

# ---------------------------------------------------------------
# Ticks 
# ---------------------------------------------------------------
def ticks_along(p_start, p_end, values, tick_dir, offset_text=(0,0,0)):
    p_start = np.array(p_start, dtype=float)
    p_end = np.array(p_end, dtype=float)
    n = len(values)
    for k, val in enumerate(values):
        t = k / (n - 1)
        p = p_start + t * (p_end - p_start)
        tick = p + np.array(tick_dir)
        ax.plot([p[0], tick[0]], [p[1], tick[1]], [p[2], tick[2]], color='0.4', lw=0.8)
        text_pos = tick + np.array(offset_text)
        ax.text(*text_pos, f'{val:g}', color='0.5', fontsize=8, ha='center', va='center')

# X-axis ticks (Front bottom edge / Ground: Screen Y=100, Screen Z=0)
ticks_along([x0, sy0, sz0], [x1, sy0, sz0], np.linspace(x0, x1, 6), (0, -10, 0), (0, 0, -25))

# Y-axis ticks (Physical Y is vertical -> Screen Z. Front left edge)
ticks_along([x0, sy1, sz1], [x0, sy1, sz0], np.linspace(sz1, sz0, 5), (-20, 10, 0), (-35, 25, 0))

# Z-axis ticks (Physical Z is depth -> Screen Y. Right bottom edge: X=500, Screen Z=0)
ticks_along([x1, sy0, sz0], [x1, sy1, sz0], np.linspace(sy0, sy1, 5), (20, 0, 0), (35, 0, 0))

# ---------------------------------------------------------------
# Inlet pipe (Cyan cylinder)
# ---------------------------------------------------------------
theta = np.linspace(0, 2 * np.pi, 40)
x_cyl = np.linspace(x0 - L, x0, 2)
theta_grid, x_grid = np.meshgrid(theta, x_cyl)
y_grid_scr = zc_scr + R * np.sin(theta_grid) # Math Z -> Screen Y
z_grid_scr = yc_scr + R * np.cos(theta_grid) # Math Y -> Screen Z

ax.plot_surface(x_grid, y_grid_scr, z_grid_scr, color='cyan', alpha=0.6, linewidth=0, shade=False)

for x_end in (x0 - L, x0):
    sy_pipe = zc_scr + R * np.sin(theta)
    sz_pipe = yc_scr + R * np.cos(theta)
    ax.plot(np.full_like(theta, x_end), sy_pipe, sz_pipe, color='k', lw=1)

# ---------------------------------------------------------------
# Triad
# ---------------------------------------------------------------

tx, ty, tz = -80, -200, -60 

axis_len = 50

# X-axis (Right / Streamwise)
ax.plot([tx, tx + axis_len +10], [ty, ty], [tz, tz], color='r', lw=1.5)
ax.text(tx + axis_len + 25, ty, tz, '$x$', color='r', fontsize=10, va='center')

# Y-axis (Physical Y is vertical -> Screen Z)
ax.plot([tx, tx], [ty, ty], [tz, tz + axis_len], color='r', lw=1.5)
ax.text(tx, ty, tz + axis_len + 15, '$y$', color='r', fontsize=10, ha='center')

# Z-axis (Physical Z is depth -> Screen Y)
ax.plot([tx, tx], [ty, ty + axis_len], [tz, tz], color='r', lw=1.5)
ax.text(tx, ty + axis_len + 15, tz, '$z$', color='r', fontsize=10, ha='center')

# ---------------------------------------------------------------
# Arrows and labels 
# ---------------------------------------------------------------
def add_arrow(start, end, text, ha='center', va='center', text_offset=(0,0,0)):
    ax.quiver(start[0], start[1], start[2],
              end[0]-start[0], end[1]-start[1], end[2]-start[2],
              color='k', arrow_length_ratio=0.15, lw=1.5)
    t_pos = np.array(start) + np.array(text_offset)
    ax.text(*t_pos, text, color='k', fontsize=10, ha=ha, va=va)

# Inlets and Outlets (X, Screen Y, Screen Z)
add_arrow([-180, zc_scr, yc_scr], [-35, zc_scr, yc_scr], 'Velocity Inlet', text_offset=(10, 0, -30))
add_arrow([x0, 50, 25], [-100, 50, 25], 'Pressure outlet', text_offset=(-190, 0, 10)) 
add_arrow([x1, 50, 175], [x1+100, 50, 175], 'Pressure outlet', text_offset=(180, 0, 0))

# ---------------------------------------------------------------
# Symmetry Lines (Anchored with a dot and angles adjusted)
# ---------------------------------------------------------------
def add_leader(start, end, text, ha='center'):
    # Add a solid dot at the starting point to "pin" it to the face
    ax.plot([start[0]], [start[1]], [start[2]], marker='o', color='k', markersize=4)
    
    # Draw the leader line
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='k', lw=1.2)
    
    # Add the text slightly above the end of the line
    ax.text(end[0], end[1], end[2] + 15, text, color='k', fontsize=10, ha=ha)

# Top face: Pinned to center (X=250), pulled straight UP
add_leader([250, 0, sz1], [250, 0, sz1 + 150], 'Symmetry')      

# Front-left face (Screen Y=-100): Pinned to center, pulled OUT toward camera
add_leader([250, sy0, 100], [200, sy0 - 280, 240], 'Symmetry')    

# Back-right face (Screen Y=100): Pinned slightly to the right, pulled BACK
add_leader([350, sy1, 100], [350, sy1 + 180, 140], 'Symmetry')

# No-Slip Wall (Ground) - Swept completely to the front-right side
t = np.linspace(0, 1, 30)
p0 = np.array([600, -140, -100])    # Text position: Far right (X=550), slightly forward, and below the box
p1 = np.array([480, -40, 0])      # Control point: Curves cleanly over the front-right edge
p2 = np.array([400, 0, 0])        # Arrow tip: Anchored on the floor near the right side

curve_x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
curve_sy = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
curve_sz = (1-t)**2 * p0[2] + 2*(1-t)*t * p1[2] + t**2 * p2[2]

ax.plot(curve_x, curve_sy, curve_sz, color='k', lw=1.5)
dir_vec = np.array([curve_x[-1]-curve_x[-2], curve_sy[-1]-curve_sy[-2], curve_sz[-1]-curve_sz[-2]])
ax.quiver(curve_x[-2], curve_sy[-2], curve_sz[-2], dir_vec[0], dir_vec[1], dir_vec[2],
          color='k', arrow_length_ratio=0.5, lw=1.5)

ax.text(p0[0], p0[1], p0[2]-15, 'No-Slip Wall (Ground)', color='k', fontsize=10, ha='center')

# ---------------------------------------------------------------
# INSET: Zoomed-in Detail of the Inlet Pipe
# ---------------------------------------------------------------
# 1. Create a new 3D axes in the top-left corner [left, bottom, width, height]
ax_ins = fig.add_axes([0.05, 0.65, 0.22, 0.22], projection='3d')

# 2. Draw a clean 2D border around the inset using figure coordinates
rect = Rectangle((0.05, 0.65), 0.22, 0.22, fill=False, color='k', lw=1.2, 
                 transform=fig.transFigure, clip_on=False)
fig.patches.append(rect)

# ---------------------------------------------------------------
# Zoom/Magnification Connection Lines
# ---------------------------------------------------------------
# These lines connect the bottom corners of the inset box to the pipe in the main plot.
# xyA = coordinates on the inset box (0 to 1)
# xyB = approximate visual coordinates on the main 3D plot (0 to 1)

zoom_line1 = ConnectionPatch(xyA=(0.0, 0.0), coordsA='axes fraction', axesA=ax_ins,
                             xyB=(0.36, 0.33), coordsB='axes fraction', axesB=ax,
                             color='0.3', linestyle='--', lw=1)

zoom_line2 = ConnectionPatch(xyA=(1.0, 0.0), coordsA='axes fraction', axesA=ax_ins,
                             xyB=(0.39, 0.34), coordsB='axes fraction', axesB=ax,
                             color='0.3', linestyle='--', lw=1)

fig.add_artist(zoom_line1)
fig.add_artist(zoom_line2)

# 3. Redraw the cyan cylinder inside the inset
ax_ins.plot_surface(x_grid, y_grid_scr, z_grid_scr, color='cyan', alpha=0.6, linewidth=0, shade=False)
for x_end in (x0 - L, x0):
    sy_pipe = zc_scr + R * np.sin(theta)
    sz_pipe = yc_scr + R * np.cos(theta)
    ax_ins.plot(np.full_like(theta, x_end), sy_pipe, sz_pipe, color='k', lw=1)

# 4. Dimension: Length (L)
z_dim = -R - 4  # Drop the line slightly below the pipe
ax_ins.plot([x0 - L, x0], [0, 0], [z_dim, z_dim], color='k', lw=1)
# Extension lines dropping down from the pipe ends
ax_ins.plot([x0 - L, x0 - L], [0, 0], [-R - 1, z_dim - 1], color='k', lw=0.8)
ax_ins.plot([x0, x0], [0, 0], [-R - 1, z_dim - 1], color='k', lw=0.8)
ax_ins.text(x0 - L/2, 0, z_dim - 2, '$L$', color='k', fontsize=12, ha='center', va='top')

# 5. Dimension: Diameter (D) - Placed safely ABOVE the pipe
z_D = R + 4  # Height of the dimension line above the pipe

# Extension lines pulling straight UP from the left and right edges of the circular face
ax_ins.plot([x0 - L, x0 - L], [-R, -R], [0, z_D + 1], color='k', lw=0.8)  
ax_ins.plot([x0 - L, x0 - L], [R, R], [0, z_D + 1], color='k', lw=0.8)    

# Main horizontal dimension line across the top
ax_ins.plot([x0 - L, x0 - L], [-R, R], [z_D, z_D], color='k', lw=1.2)     

# The text sits cleanly on top of the dimension line, just saying 'D'
ax_ins.text(x0 - L, 0, z_D + 1, '$D$', color='k', fontsize=12, ha='center', va='bottom')

# 6. Format the inset viewport
ax_ins.set_xlim(x0 - L - 5, x0 + 5)
ax_ins.set_ylim(-R - 5, R + 5)
ax_ins.set_zlim(z_dim - 2, z_D + 4) # Expanded to fit L below and D above

# Dynamically lock aspect ratio so the pipe doesn't stretch
dx = (x0 + 5) - (x0 - L - 5)
dy = (R + 5) - (-R - 5)
dz = (z_D + 4) - (z_dim - 2)
ax_ins.set_box_aspect((dx, dy, dz))  

ax_ins.view_init(elev=20, azim=-40)                  
ax_ins.set_axis_off()

fig.text(0.05, 0.88, "Detail A: Inlet Pipe", fontsize=11, fontweight='bold', fontfamily='serif')

# Final adjustments
ax.set_axis_off()
ax.set_proj_type('ortho')
fig.subplots_adjust(left=0.05, right=1.05, bottom=-0.25, top=1.1)

save_dir = r"C:\Users\arjun\OneDrive - Cranfield University\Documents\IRP\manuscript_template\manuscript_example"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'Figure01_domain.png'), dpi=300)

plt.show()