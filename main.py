import os, sys
import subprocess
import time
import zipfile
import shutil
import requests
from tqdm import tqdm  # For progress tracking

    
from datetime import datetime
import threading
from IPython.core.getipython import get_ipython

def zip_folder(folder_path, zip_path):
    """Zip the contents of a folder."""
    print("="*60, "Starting to zip folder...", "-"*60, sep="\n")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    print("="*60, "Folder zipped successfully.", "-"*60, sep="\n")

def move_files(source_folder, dest_folder):
    """Move files from source folder to destination folder."""
    print("="*60, "Starting to move files...", "-"*60, sep="\n")
    os.makedirs(dest_folder, exist_ok=True)
    if os.path.exists(source_folder):
        for item in os.listdir(source_folder):
            try:
                shutil.move(os.path.join(source_folder, item), dest_folder)
            except Exception as e:
                print(f"Error moving file {item}: {e}")
    print("="*60, "Files moved successfully.", "-"*60, sep="\n")


def install_package(url):
    """Install a package from a given URL."""
    try:
        print("="*60, f"Starting to install package from {url}...", "-"*60, sep="\n")
        package_name = url.split('/')[-1].replace('.git', '')
        package_path = f"/kaggle/working/ComfyUI/custom_nodes/{package_name}"

        if not os.path.exists("/kaggle/working/ComfyUI/custom_nodes/"):
            os.makedirs("/kaggle/working/ComfyUI/custom_nodes/")

        os.chdir("/kaggle/working/ComfyUI/custom_nodes")

        if not os.path.exists(package_path):
            subprocess.run(['git', 'clone', url, '--recursive'], check=True, text=True, capture_output=True)
            print(f"Package {url} path newly git cloned")
        else:
            os.chdir(package_path)
            subprocess.run(['git', 'pull', '--all'], check=True, text=True, capture_output=True)
            print(f"Package {url} path exist already, just updated")

        if os.path.exists("requirements.txt"):
            subprocess.run(['uv','pip', 'install', '--system', '-r', 'requirements.txt', '--quiet'], check=True, text=True, capture_output=True)

        if os.path.exists("install.py"):
            subprocess.run([sys.executable,'install.py'], check=True, text=True, capture_output=True)

        print("="*60, f"Package from {url} installed successfully.", "-"*60, sep="\n")
    except Exception as e:
        print(f"An error occurred while installing package from {url}: {str(e)}")
        # You can add additional error handling or logging here if needed

def setup_comfyui():
    """Set up ComfyUI and install necessary packages."""
    print("="*60, "Starting to set up ComfyUI...", "-"*60, sep="\n")
    os.chdir("/kaggle/working")
    if not os.path.exists("/kaggle/working/ComfyUI/") or not os.path.exists("/kaggle/working/ComfyUI/requirements.txt") :
        subprocess.run(['git', 'clone', 'https://github.com/comfyanonymous/ComfyUI.git', '--recursive'], check=True, text=True, capture_output=True)
    os.chdir("ComfyUI")
    subprocess.run(['git', 'pull', '--all'])
    subprocess.run(['uv','pip', 'install', '--system', '-r', 'requirements.txt', '--quiet'])

    model_dirs = [
        "/kaggle/working/ComfyUI/models/LLM",
        "/kaggle/working/ComfyUI/models/pulid",
        "/kaggle/working/ComfyUI/models/clip_vision",
        "/kaggle/working/ComfyUI/models/xlabs/contronets/"
        "/kaggle/working/ComfyUI/models/InstantIR/models/"
    ]
    for dir_path in model_dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("="*60, "ComfyUI set up successfully.", "-"*60, sep="\n")

def xlinkthis(src, dest):
    import os
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest) and not os.path.islink(dest):
        os.symlink(src, dest)

def link_models():
    """Create symbolic links for model files."""
    print("="*60, "Starting to link models...", "-"*60, sep="\n")
    links = [
        ("/kaggle/input/clip-gmp-vit-l-14/tensorflow2/default/1/model.safetensors",
         "/kaggle/working/ComfyUI/models/clip/CLIP_gmp_VIT_L_14.safetensors"),
        ("/kaggle/input/t5xxl_fp16.safetensors/tensorflow2/default/1/t5xxl_fp16.safetensors",
         "/kaggle/working/ComfyUI/models/clip/t5xxl_fp16.safetensors"),
        ("/kaggle/input/flux-vae-safetensors/pytorch/default/1/ae.safetensors",
         "/kaggle/working/ComfyUI/models/vae/FLuX_ae.safetensors"),
        ("/kaggle/input/florence2/transformers/default/1/Florence2_model_3/model.safetensors",
         "/kaggle/working/ComfyUI/models/LLM/Florence_model_3.safetensors"),
        ("/kaggle/input/florence2/transformers/default/1/Florence2_model_2/model.safetensors",
         "/kaggle/working/ComfyUI/models/LLM/Florence_model_2.safetensors"),
        ("/kaggle/input/controlnet/pytorch/flux-canny-controlnet/1/flux-canny-controlnet-v3.safetensors",
         "/kaggle/working/ComfyUI/models/xlabs/contronets/flux-canny-controlnet-v3.safetensors"),
        ("/kaggle/input/controlnet/pytorch/flux-depth-controlnet/1/flux-depth-controlnet-v3.safetensors",
         "/kaggle/working/ComfyUI/models/xlabs/contronets/flux-depth-controlnet-v3.safetensors"),
        # ("/kaggle/input/realvisxl-v50/pytorch/default/1/realvisxl-v50.safetensors",
        #  "/kaggle/working/ComfyUI/models/checkpoints/realvisxl-v50.safetensors"),
        # ("/kaggle/input/realvisxl-v50/pytorch/default/2/SG161222_RealVisXL_V5_0_Lightning.safetensors",
        #  "/kaggle/working/ComfyUI/models/checkpoints/SG161222_RealVisXL_V5_0_Lightning.safetensors"),
        ("/kaggle/input/flux1devnf46stepsnsfw/pytorch/default/1/flux1DevNF46StepsNSFW_fluxdevFP86Steps.safetensors",
         "/kaggle/working/ComfyUI/models/checkpoints/flux1DevNF46StepsNSFW_fluxdevFP86Steps.safetensors"),
        ("/kaggle/input/8stepscrearthyperfluxdevbnb_v24hyperdevfp8unet_/pytorch/default/1/8StepsCreartHyperFluxDevBnb_v24HyperDevFp8Unet.safetensors",
         "/kaggle/working/ComfyUI/models/unet/8StepsCreartHyperFluxDevBnb_v24HyperDevFp8Unet.safetensors"),
        ("/kaggle/input/creart-hyper-flux-dev-gguf-q4_0/gguf/default/1/CreArt-Hyper-Flux-Dev-gguf-q4_0.gguf",
         "/kaggle/working/ComfyUI/models/unet/CreArt-Hyper-Flux-Dev-gguf-q4_0.gguf"),
        ("/kaggle/input/unet/gguf/flux1-dev-q4_k_s/1/flux1-dev-Q4_K_S.gguf",
         "/kaggle/working/ComfyUI/models/unet/flux1-dev-Q4_K_S.gguf"),
        ("/kaggle/input/aldebodobasexl_v3mini/tensorflow2/default/1/albedobaseXL_v3Mini.safetensors",
         "/kaggle/working/ComfyUI/models/checkpoints/albedobaseXL_v3Mini.safetensors"),
        ("/kaggle/input/alimama-flux-turbo-lora/pytorch/default/1/diffusion_pytorch_model.safetensors",
         "/kaggle/working/ComfyUI/models/loras/alimama-flux-turbo-lora.safetensors"),
        ("/kaggle/input/fluxartfusion4steps_v12/pytorch/default/1/fluxArtfusion4Steps_v12.safetensors",
         "/kaggle/working/ComfyUI/models/checkpoints/fluxArtfusion4Steps_v12.safetensors"),
        ("/kaggle/input/clip-vit/pytorch/h-14-laion2b-s32b-b79k/1/pytorch_model.bin",
         "/kaggle/working/ComfyUI/models/clip_vision/clip_vit_h-14-laion2b-s32b-b79k.bin"),
        ("/kaggle/input/controlnet-package-hsbd10/pytorch/default/4/controlnet_instantid_model.safetensors",
         "/kaggle/working/ComfyUI/models/controlnet/instantid/LeInstantID_ControlNet_model.safetensors"),
        ("/kaggle/input/instant_id/pytorch/default/2/instantid-ip-adapter.bin",
         "/kaggle/working/ComfyUI/models/instantid/InstantID_ip_adapter.bin"),
        ("/kaggle/input/instant_id/pytorch/default/2/instantid-ip-adapter.bin",
         "/kaggle/working/ComfyUI/models/instantid/ip_adapter.bin"),
        ("/kaggle/input/instant_id/pytorch/default/2/instantid-ip-adapter.bin",
         "/kaggle/working/ComfyUI/models/instantid/SDXL/ip-adapter.bin"),
        ("/kaggle/input/controlnet-package-hsbd10/pytorch/default/4/controlnet_instantid_model.safetensors",
         "/kaggle/working/ComfyUI/models/controlnet/instantid/LeInstantID_ControlNet_model.safetensors"),
        ("/kaggle/input/4xnomosunidat_upscaler/pytorch/default/5/4xnomosunidat_upscaler.safetensors",
         "/kaggle/working/ComfyUI/models/upscale_models/4xnomosunidat_upscaler.safetensors")
        ,("/kaggle/input/flashface/pytorch/default/1/flashface.ckpt",
        "/kaggle/working/ComfyUI/models/flashface/flashface.ckpt")
    ]
    for src, dest in links:
        xlinkthis(src,dest)
    print("="*60, "Models linked successfully.", "-"*60, sep="\n")

def link_additional_models():
    """Link additional models from specified directories."""
    print("="*60, "Starting to link additional models...", "="*60, sep="\n")

    # Link models from /kaggle/input/4xnomosunidat_upscaler/pytorch/default/5/
    for x in os.listdir("/kaggle/input/4xnomosunidat_upscaler/pytorch/default/5/"):
        print(x)
        xlinkthis(f"/kaggle/input/4xnomosunidat_upscaler/pytorch/default/5/{x}", f"/kaggle/working/ComfyUI/models/upscale_models/{x}")

    # Link models from /kaggle/input/t5_xxl_gguf_models/pytorch/default/1/
    for x in os.listdir("/kaggle/input/t5_xxl_gguf_models/pytorch/default/1/"):
        print(x)
        xlinkthis(f"/kaggle/input/t5_xxl_gguf_models/pytorch/default/1/{x}", f"/kaggle/working/ComfyUI/models/clip/{x}")

    # Link models from /kaggle/input/sdxl_controlnets/pytorch/default/3/
    for x in os.listdir("/kaggle/input/sdxl_controlnets/pytorch/default/3/"):
        print(x)
        xlinkthis(f"/kaggle/input/sdxl_controlnets/pytorch/default/3/{x}", f"/kaggle/working/ComfyUI/models/controlnet/{x}")
        if 'ip-adapter' in x:
            xlinkthis(f"/kaggle/input/sdxl_controlnets/pytorch/default/3/{x}", f"/kaggle/working/ComfyUI/models/ipadapter/{x}")

    # Link models from /kaggle/input/pulid-model/pytorch/default/3/
    for x in os.listdir("/kaggle/input/pulid-model/pytorch/default/3/"):
        print(x)
        xlinkthis(f"/kaggle/input/pulid-model/pytorch/default/3/{x}", f"/kaggle/working/ComfyUI/models/pulid/{x}")

    for x in os.listdir("/kaggle/input/instant_id/pytorch/default/2"):
        print(x)
        xlinkthis(f"/kaggle/input/instant_id/pytorch/default/2/{x}", f"/kaggle/working/ComfyUI/models/controlnet/{x}")
        #/kaggle/input/instant_id/pytorch/default/2
    
    for x in os.listdir("/kaggle/input/controlnet-package-hsbd10/pytorch/default/4/"):
        print(x)
        xlinkthis(f"/kaggle/input/controlnet-package-hsbd10/pytorch/default/4/{x}", f"/kaggle/working/ComfyUI/models/controlnet/{x}")
    
    
    for x in os.listdir("/kaggle/input/dmd2/pytorch/default/1"):
        print(x)
        if 'lora' in x:
            xlinkthis(f"/kaggle/input/dmd2/pytorch/default/1/{x}", f"/kaggle/working/ComfyUI/models/loras/{x}")
        elif 'unet' in x.lower():
            xlinkthis(f"/kaggle/input/dmd2/pytorch/default/1/{x}", f"/kaggle/working/ComfyUI/models/unet/{x}")
        else:
            xlinkthis(f"/kaggle/input/dmd2/pytorch/default/1/{x}", f"/kaggle/working/ComfyUI/models/checkpoints/{x}")

    # /kaggle/input/juggernaut/pytorch/default/1

    for x in os.listdir("/kaggle/input/juggernaut/pytorch/default/1/"):
        print(x)
        xlinkthis(f"/kaggle/input/juggernaut/pytorch/default/1/{x}", f"/kaggle/working/ComfyUI/models/checkpoints/{x}")

    #SDXl-mODELS
    for x in os.listdir("/kaggle/input/realvisxl-v50/pytorch/default/3/"):
        print(x)
        xlinkthis(f"/kaggle/input/realvisxl-v50/pytorch/default/3/{x}", f"/kaggle/working/ComfyUI/models/checkpoints/{x}")

    #InstantIR
    for x in os.listdir("/kaggle/input/instantir/pytorch/default/2/"):
        print(x)
        xlinkthis(f"/kaggle/input/instantir/pytorch/default/2/{x}", f"/kaggle/working/ComfyUI/models/InstantIR/models/{x}")
    
    source_dir = "/kaggle/input/nix-flux-fusion-models/pytorch/default/1/"
    destination_dir = "/kaggle/working/ComfyUI/models/unet/"
    # NIX-FLUX-FUSION-MODELS : Iterate through files in the source directory
    for x in os.listdir(source_dir):
        source_path = os.path.join(source_dir, x)
        destination_path = os.path.join(destination_dir, x)
        print(x)
        xlinkthis(source_path,destination_path)


    
    
    # /kaggle/input/controlnet-package-hsbd10/pytorch/default/4
    print("="*60, "Additional models linked successfully.", "="*60, sep="\n")

def start_comfyui_instances():
    """Start ComfyUI instances."""
    print("="*60, "Starting ComfyUI instances...", "-"*60, sep="\n")
    os.chdir("/kaggle/working/ComfyUI/")
    # xP1 = subprocess.Popen([sys.executable, "main.py", "--cuda-device", "0", "--port", "8188", "--fp8_e4m3fn-text-enc","--fp8_e4m3fn-unet", "--highvram"]) #--fp8_e4m3fn-text-enc --fp8_e4m3fn-unet
    os.makedirs("/kaggle/working/ComfyUI/logs", exist_ok=True)
    with open("/kaggle/working/ComfyUI/logs/comfy_8188.log", "a") as logfile:
        xP1 = subprocess.Popen(
            [sys.executable, "main.py", "--cuda-device", "0", "--port", "8188", 
             "--fp8_e4m3fn-text-enc", "--fp8_e4m3fn-unet", "--highvram"],
            stdout=logfile,
            stderr=subprocess.STDOUT
        )
    time.sleep(10)
    with open("/kaggle/working/ComfyUI/logs/comfy_8189.log", "a") as logfile:
        xP2 = subprocess.Popen([sys.executable, "main.py", "--cuda-device", "1", "--port", "8189", "--fp8_e4m3fn-text-enc","--fp8_e4m3fn-unet", "--highvram"]
                                          ,stdout=logfile, stderr=subprocess.STDOUT) #, "--highvram"])
    time.sleep(10)
    print("="*60, "ComfyUI instances started successfully.", "-"*60, sep="\n")

def start_playit_agent():
    """Start the Playit agent."""
    print("="*60, "Starting Playit agent...", "-"*60, sep="\n")
    os.chdir("/kaggle/working/ComfyUI/")
    if not os.path.exists('/kaggle/working/ComfyUI/playit-linux-amd64'):
        subprocess.run(['wget', 'https://github.com/playit-cloud/playit-agent/releases/download/v0.15.26/playit-linux-amd64'])
    subprocess.run(['chmod', '+x', './playit-linux-amd64'])
    os.makedirs("/kaggle/working/ComfyUI/logs", exist_ok=True)
    subprocess.Popen("nohup /kaggle/working/ComfyUI/playit-linux-amd64 >> /kaggle/working/ComfyUI/logs/playit.log 2>&1 &", bufsize=0, shell=True)
    time.sleep(10)
    xresult = subprocess.run(['tail', '-n', '20', '/kaggle/working/ComfyUI/logs/playit.log'] ,capture_output=True,text=True)
    print(xresult.stdout,xresult.stderr,sep='\n'+'.'*10)
    print("","="*60, "Playit agent started successfully.", "-"*60, sep="\n")

def execute_first_cell():
    """Execute the first cell in the notebook."""
    print("="*60, "Executed", datetime.today(), "-"*60, sep="\n")
    get_ipython().run_cell(get_ipython().user_ns['In'][1])
    print("="*60, "-"*60, sep="\n")

def scheduler():
    """Scheduler to execute the first cell every minute."""
    print("="*60, "Starting scheduler...", "-"*60, sep="\n")
    while True:
        execute_first_cell()
        time.sleep(60)  # 60 seconds
# ===============================================
# def down_antelope():
#     # Define the URL and the target directory
#     url = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
#     target_dir = "/kaggle/working/ComfyUI/models/insightface/models/antelopev2/"
    
#     # Create target directory if it doesn't exist
#     os.makedirs(target_dir, exist_ok=True)
    
#     # Download the ZIP file
#     response = requests.get(url)
#     zip_file_path = os.path.join(target_dir, "antelopev2.zip")
    
#     with open(zip_file_path, "wb") as f:
#         f.write(response.content)
    
#     # Unzip the file
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         zip_ref.extractall(target_dir)
    
#     # Optionally, remove the ZIP file after extraction
#     os.remove(zip_file_path)
    
#     print("Download and extraction completed for Antelopev2.")
# ===============================================

def down_antelope():
    # Define the URL and the target directory
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
    target_dir = "/kaggle/working/ComfyUI/models/insightface/models/"
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Define the path for the downloaded ZIP file
    zip_file_path = os.path.join(target_dir, "antelopev2.zip")
    
    try:
        # Download the ZIP file with progress tracking
        print("Downloading Antelopev2 model...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_file_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                progress_bar.update(size)
        
        print("Download completed. Extracting files...")
        
        # Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        # Optionally, remove the ZIP file after extraction
        os.remove(zip_file_path)
        
        print("Extraction completed. Antelopev2 model is ready.")
    
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error during extraction: {e}. The downloaded file might be corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# ===============================================
# def down_landmark():
#     subprocess.Popen("apt install aria2 -qq", bufsize=0, shell=True)
#     subprocess.Popen("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/bluefoxcreation/FaceAlignment/resolve/main/fan2_68_landmark.onnx?download=true -d /kaggle/working/ComfyUI/models/landmarks -o fan2_68_landmark.onnx", bufsize=0, shell=True)
#     subprocess.Popen("aria2c  --console-log-level=error -c  -x16 -s16 -j5  -k 1M 'https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download'  -d /kaggle/working/ComfyUI/models/bisenet -o 79999_iter.pth", bufsize=0, shell=True)
#     subprocess.Popen("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt?download=true  -d /kaggle/working/ComfyUI/models/ultralytics/bbox -o face_yolov8m.pt", bufsize=0, shell=True)
#     subprocess.Popen("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors?download=true -d /kaggle/working/ComfyUI/models/controlnet/ -o control_v11p_sd15_inpaint_fp16.safetensors", bufsize=0, shell=True)
#     print("Download and extraction completed for landmark and bbox and sd15 inpaint.")

# ===============================================
def down_landmark():
    # Install aria2 quietly
    print("Installing aria2...")
    try:
        subprocess.run("apt install aria2 -y -qq", shell=True, check=True)
        print("aria2 installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install aria2: {e}")
        return

    # Define the list of downloads
    downloads = [
        {
            "url": "https://huggingface.co/bluefoxcreation/FaceAlignment/resolve/main/fan2_68_landmark.onnx?download=true",
            "dir": "/kaggle/working/ComfyUI/models/landmarks",
            "output": "fan2_68_landmark.onnx"
        },
        {
            "url": "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download",
            "dir": "/kaggle/working/ComfyUI/models/bisenet",
            "output": "79999_iter.pth"
        },
        {
            "url": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt?download=true",
            "dir": "/kaggle/working/ComfyUI/models/ultralytics/bbox",
            "output": "face_yolov8m.pt"
        },
        {
            "url": "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors?download=true",
            "dir": "/kaggle/working/ComfyUI/models/controlnet/",
            "output": "control_v11p_sd15_inpaint_fp16.safetensors"
        }

    ]

    # Download each file sequentially
    for i, download in enumerate(downloads):
        url = download["url"]
        dir_path = download["dir"]
        output_file = download["output"]

        print(f"Downloading file {i + 1}/{len(downloads)}: {output_file}...")
        try:
            subprocess.run(
                f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M '{url}' -d '{dir_path}' -o '{output_file}'",
                shell=True,
                check=True
            )
            print(f"Download completed for {output_file}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {output_file}: {e}")

    xlist = """https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/reswapper_256.onnx?download=true
    https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/reswapper_128.onnx?download=true
    https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx?download=true""".split('\n')
    
    os.makedirs("/kaggle/working/ComfyUI/models/reswapper/",exist_ok=True)
    
    for i,x in enumerate(xlist):
        print(i,'-',x)
        filnm = x.split('/')[-1].replace('?download=true','')
        print(filnm)
        # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $x -d /kaggle/working/ComfyUI/models/reswapper/ -o $filnm
        print(f"Downloading file {i + 1}/{len(downloads)}: {output_file}...")
        try:
            subprocess.run(
                f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M '{x}' -d '/kaggle/working/ComfyUI/models/reswapper/' -o '{filnm}'",
                shell=True,
                check=True
            )
            print(f"Download completed for {filnm}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {filnm}: {e}")
    print("All downloads completed.")
    
def install_packages_list():
    # Install packages
    print("Start Install packages list ... ")
    packages = [
        "https://github.com/comfyanonymous/ComfyUI_bitsandbytes_NF4",
        "https://github.com/WASasquatch/was-node-suite-comfyui",
        "https://github.com/rgthree/rgthree-comfy",
        "https://codeberg.org/Gourieff/comfyui-reactor-node",
        "https://github.com/BlenderNeko/ComfyUI_Noise",
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "https://github.com/ltdrdata/ComfyUI-Impact-Subpack",
        
        "https://github.com/yolain/ComfyUI-Easy-Use",
        "https://github.com/kijai/ComfyUI-KJNodes",
        "https://github.com/kijai/ComfyUI-Florence2",
        "https://github.com/XLabs-AI/x-flux-comfyui",
        "https://github.com/crystian/ComfyUI-Crystools",
        "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes",
        "https://github.com/city96/ComfyUI-GGUF",
        "https://github.com/cubiq/PuLID_ComfyUI",
        "https://github.com/cubiq/ComfyUI_InstantID",
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        "https://github.com/comfyanonymous/ComfyUI_experiments",
        "https://github.com/chengzeyi/Comfy-WaveSpeed",
        "https://github.com/Fannovel16/comfyui_controlnet_aux",
        "https://github.com/ShmuelRonen/ComfyUI-FreeMemory",
        "https://github.com/jamesWalker55/comfyui-various",
        "https://github.com/kijai/ComfyUI-SUPIR",
        "https://github.com/klinter007/klinter_nodes",
        "https://github.com/holchan/ComfyUI-ModelDownloader",
        "https://github.com/KoreTeknology/ComfyUI-Universal-Styler",
        "https://github.com/marduk191/ComfyUI-Fluxpromptenhancer",
        "https://github.com/fairy-root/Flux-Prompt-Generator",
        "https://github.com/VykosX/ControlFlowUtils",
        "https://github.com/godmt/ComfyUI-List-Utils",
        "https://github.com/pythongosssss/ComfyUI-WD14-Tagger",
        "https://github.com/miaoshouai/ComfyUI-Miaoshouai-Tagger",
        "https://github.com/GraftingRayman/ComfyUI_GraftingRayman",
        "https://github.com/pollockjj/ComfyUI-MultiGPU",
        "https://github.com/dchatel/comfyui_facetools",
        "https://github.com/fssorc/ComfyUI_FaceShaper",
        
        "https://github.com/vuongminh1907/ComfyUI_ZenID",
        "https://github.com/nuanarchy/ComfyUI-NuA-FlashFace",
        "https://github.com/nosiu/comfyui-instantId-faceswap",
        "https://github.com/EnragedAntelope/ComfyUI-EACloudNodes",
        "https://github.com/nicofdga/DZ-FaceDetailer",
        "https://github.com/mav-rik/facerestore_cf",
        "https://github.com/smthemex/ComfyUI_InstantIR_Wrapper",
        "https://github.com/cubiq/ComfyUI_essentials",

        
        "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet",
        "https://github.com/EvilBT/ComfyUI_SLK_joy_caption_two",
        "https://github.com/kijai/ComfyUI-LivePortraitKJ",
        "https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait",
        "https://github.com/ssitu/ComfyUI_UltimateSDUpscale",
        "https://github.com/kijai/ComfyUI-CCSR",
        "https://github.com/TheBill2001/comfyui-upscale-by-model",
        "https://github.com/zentrocdot/ComfyUI-RealESRGAN_Upscaler",
        "https://github.com/traugdor/ComfyUI-UltimateSDUpscale-GGUF",
        "https://github.com/jags111/efficiency-nodes-comfyui",
        "https://github.com/kijai/ComfyUI-segment-anything-2",
        "https://github.com/Extraltodeus/ComfyUI-AutomaticCFG",
        "https://github.com/ltdrdata/ComfyUI-Inspire-Pack",
        "https://github.com/evanspearman/ComfyMath",
        "https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes",
        "https://github.com/storyicon/comfyui_segment_anything",
        "https://github.com/PrunaAI/ComfyUI_pruna",

        "https://github.com/zhangp365/ComfyUI-utils-nodes",
        "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
        "https://github.com/revirevy/Comfyui_saveimage_imgbb",
        "https://github.com/ltdrdata/ComfyUI-Manager",
        
        ]
    for package in packages:
        install_package(package)
        
    print(f"All Packages list installed \n {'='*10}  DONE  {'='*10}")

def main():
    """Main function to orchestrate the setup and execution."""
    print("="*60, "Starting main function...", "-"*60, sep="\n")
    # Install uv
    print("="*60, "Installing uv...", "-"*60, sep="\n")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 'pip', 'uv', '-q'], check=True, text=True, capture_output=True)
    # subprocess.run(['uv','pip', 'install', '--system', 'sageattention', '--quiet'])

    # Example usage of zip_folder
    folder_to_zip = '/kaggle/working/ComfyUI/output'
    if os.path.exists(folder_to_zip):
        output_zip = f'/kaggle/working/output_{datetime.now():%d%m%Y_%H%M}.zip'
        zip_folder(folder_to_zip, output_zip)

        # Move files
        x_old = '/kaggle/working/old_output'
        move_files(folder_to_zip, x_old)
        
    # Link models
    link_models()

    # link_additional_models
    link_additional_models()
    
    # Setup ComfyUI
    setup_comfyui()

    # Install packages
    install_packages_list()

    # Setup ComfyUI
    setup_comfyui()


    # Start ComfyUI instances
    start_comfyui_instances()

    # Start Playit agent
    start_playit_agent()

    # download antelope
    down_antelope()

    down_landmark()   

    # Start scheduler
    scheduler_thread = threading.Thread(target=scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

    print("="*60, "Scheduler started in background. It will execute the first cell every 1 minute.", "-"*60, sep="\n")

if __name__ == "__main__":
    main()
