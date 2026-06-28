from IPython.display import Image, display
import git
import os
import subprocess
import glob
import ultralytics
import torch
from ultralytics import YOLO

os.chdir('C:\\Users\\Lunafernandavid\\Documents\\Documentos Propios Juan David\\Maestria manizales\\Carpeta BIOS')
HOME = os.getcwd()

print(HOME)

ultralytics.checks()

os.makedirs(f'{HOME}/datasets', exist_ok=True)

os.chdir(f'{HOME}/datasets')


#!pip install roboflow

#from roboflow import Roboflow
#rf = Roboflow(api_key="IiFztbTBnwLTHykDgvdl")
#project = rf.workspace("datoscancer").project("datos_cancer_poligono-apnys")
#version = project.version(1)
#dataset = version.download("yolov8")
                

def main():
    # Definir directorios y configuraciones necesarias
    data_directory = 'C:\\Users\\Lunafernandavid\\Documents\\Documentos Propios Juan David\\Maestria manizales\\Carpeta BIOS\\datasets\\Datos_Cancer_poligono-1'
    modelo_guardado = 'C:\\Users\\Lunafernandavid\\Documents\\Documentos Propios Juan David\\Maestria manizales\\Carpeta BIOS\\modelo_guardado\\modelo_guardado.pt'
    
    # Cambiar al directorio de trabajo donde están los datos
    os.chdir(data_directory)
    
    # Verificar si CUDA está disponible y seleccionar el dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Liberar memoria antes de cargar el modelo
    torch.cuda.empty_cache()
    
    # Cargar el modelo YOLOv8 y moverlo a la GPU si está disponible
    model = YOLO('yolov8n.pt').to(device)
    
    # Ajustar configuraciones de memoria y entrenar el modelo
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.8, 0)

    # Crear el optimizador Adam
    optimizer='SGD'
    
    # Entrenar el modelo con un tamaño de lote reducido
    model.train(
        data='C:\\Users\\Lunafernandavid\\Documents\\Documentos Propios Juan David\\Maestria manizales\\Carpeta BIOS\\datasets\\Datos_Cancer_poligono-1\\data.yaml',
        epochs=3,
        imgsz=640,
        batch=8,
        multi_scale=True,
        device=device,
        optimizer=optimizer
    )
    
    # Guardar los pesos manualmente después de entrenar
    model.save(modelo_guardado)

if __name__ == '__main__':
    main()