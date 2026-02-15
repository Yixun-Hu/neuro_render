import open3d as o3d
import numpy as np

filename = "output/lego/train/ours_30000/fuse_post.ply"

# Load as triangle mesh (not point cloud)
mesh = o3d.io.read_triangle_mesh(filename)
mesh.compute_vertex_normals()

print(f"Vertices: {len(mesh.vertices)}")
print(f"Triangles: {len(mesh.triangles)}")
print(f"Has vertex colors: {mesh.has_vertex_colors()}")

# Render from multiple angles and save images
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=800, visible=False)
vis.add_geometry(mesh)

# Set rendering options
opt = vis.get_render_option()
opt.mesh_show_back_face = True
opt.background_color = np.array([1, 1, 1])  # white background

# Render front view
ctr = vis.get_view_control()
ctr.set_zoom(0.7)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("output/lego/train/ours_30000/mesh_view1.png")

# Rotate and render side view
ctr.rotate(400.0, 0.0)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("output/lego/train/ours_30000/mesh_view2.png")

# Another angle
ctr.rotate(400.0, 200.0)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("output/lego/train/ours_30000/mesh_view3.png")

vis.destroy_window()
print("Saved mesh_view1.png, mesh_view2.png, mesh_view3.png")
