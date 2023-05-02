import cv2
import os

def anim_wigner(density = True, surface = False):
    path = os.path.join(os.getcwd(),'wigner_frames')
    
    if density:
        # Get images in each folder
        density_files_unsorted = [os.path.join(path,f) for f in os.listdir(path) if (f.endswith('.png') and f.startswith('density'))]
        # Sort frames in correct order (0,1,2...10,11) instead of (0,1,10,11..2,20)  
        density_dict = {}

        for img_file in density_files_unsorted:
            frame_number = int(os.path.basename(str(img_file)).split('_')[-1].split('.')[0])
            density_dict[frame_number] = img_file

        density_files = [value for key,value in sorted(density_dict.items())]

        # Create video writers
        fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        density_frame_size = cv2.imread(density_files[0]).shape[:2]
        
        # Reverse shape tuple order from (height,width) to (width,height)
        density_frame_size = density_frame_size[::-1]
        
        density_out = cv2.VideoWriter('density.mp4', fourcc, fps, density_frame_size)
        
        # Loop through each frame
        for i in range(len(density_files)):
            density_img = cv2.imread(density_files[i])
            density_out.write(density_img)
        
        density_out.release()
        cv2.destroyAllWindows()
    
    if surface: 
        surface_files_unsorted = [os.path.join(path,f) for f in os.listdir(path) if (f.endswith('.png') and f.startswith('3d'))]
        surface_dict = {}

        for img_file in surface_files_unsorted:
            frame_number = int(os.path.basename(str(img_file)).split('_')[-1].split('.')[0])
            surface_dict[frame_number] = img_file

        surface_files = [value for key,value in sorted(surface_dict.items())]

        fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
        surface_frame_size = cv2.imread(surface_files[0]).shape[:2]

        surface_frame_size = surface_frame_size[::-1]

        surface_out = cv2.VideoWriter('3d.mp4', fourcc, fps, surface_frame_size)

        for i in range(len(surface_files)):
            surface_img = cv2.imread(surface_files[i])
            surface_out.write(surface_img)
    
        surface_out.release()
        cv2.destroyAllWindows()