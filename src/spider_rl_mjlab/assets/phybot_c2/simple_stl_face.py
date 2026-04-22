import os
import shutil
import trimesh

input_dir = "/home/jc/phybot_c1_v2/phybot_c2/meshes"
output_dir = "/home/jc/phybot_c1_v2/phybot_c2/new_meshes"
threshold_faces = 200000  
target_faces = 100000     

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith('.stl'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            scene_or_mesh = trimesh.load(input_path)

            if isinstance(scene_or_mesh, trimesh.Scene):

                mesh = scene_or_mesh.dump(concatenate=True)
            else:
                mesh = scene_or_mesh
            
            current_faces = len(mesh.faces)

            if current_faces > threshold_faces:
                target = min(target_faces, current_faces)
                simplified = mesh.simplify_quadric_decimation(face_count=target)
                simplified.export(output_path)
            else:
                shutil.copy2(input_path, output_path)

            check_mesh = trimesh.load(output_path)
            if isinstance(check_mesh, trimesh.Scene):
                check_mesh = check_mesh.dump(concatenate=True)
  
        except Exception as e:
            continue

