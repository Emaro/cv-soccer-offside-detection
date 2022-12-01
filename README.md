# cv-soccer-offside-detection

# Setting Up Environments for YOLOv5

conda create -n env_name
conda activate env_name
conda install pytorch-cpu torchvision-cpu -c pytorch
pip install -r yolov5/requirements.txt

# How to RUN
cd yolov5
python detect_coordinates.py 
