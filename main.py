import os
import subprocess
import time
import zipfile
import shutil
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
    print("="*60, f"Starting to install package from {url}...", "-"*60, sep="\n")
    package_name = url.split('/')[-1].replace('.git', '')
    package_path = f"/kaggle/working/ComfyUI/custom_nodes/{package_name}"

    if not os.path.exists("/kaggle/working/ComfyUI/custom_nodes/"):
        os.makedirs("/kaggle/working/ComfyUI/custom_nodes/")

    os.chdir("/kaggle/working/ComfyUI/custom_nodes")

    if not os.path.exists(package_path):
        get_ipython().system(f'git clone {url} --recursive')

    os.chdir(package_path)
    get_ipython().system('git pull --all')

    if os.path.exists("requirements.txt"):
        get_ipython().system('uv pip install --system -r requirements.txt --quiet')

    print("="*60, f"Package from {url} installed successfully.", "-"*60, sep="\n")

def setup_comfyui():
    """Set up ComfyUI and install necessary packages."""
    print("="*60, "Starting to set up ComfyUI...", "-"*60, sep="\n")
    os.chdir("/kaggle/working")
    if not os.path.exists("/kaggle/working/ComfyUI/") or not os.path.exists("/kaggle/working/ComfyUI/requirements.txt") :
        get_ipython().system('git clone https://github.com/comfyanonymous/ComfyUI.git --recursive')
    os.chdir("ComfyUI")
    get_ipython().system('git pull --all')
    get_ipython().system('uv pip install --system -r requirements.txt --quiet')

    model_dirs = [
        "/kaggle/working/ComfyUI/models/LLM",
        "/kaggle/working/ComfyUI/models/pulid",
        "/kaggle/working/ComfyUI/models/clip_vision",
        "/kaggle/working/ComfyUI/models/xlabs/contronets/"
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
        ("/kaggle/input/realvisxl-v50/pytorch/default/1/realvisxl-v50.safetensors",
         "/kaggle/working/ComfyUI/models/checkpoints/realvisxl-v50.safetensors"),
        ("/kaggle/input/realvisxl-v50/pytorch/default/2/SG161222_RealVisXL_V5_0_Lightning.safetensors",
         "/kaggle/working/ComfyUI/models/checkpoints/SG161222_RealVisXL_V5_0_Lightning.safetensors"),
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
        os.symlink(f"/kaggle/input/sdxl_controlnets/pytorch/default/3/{x}", f"/kaggle/working/ComfyUI/models/controlnet/{x}")
        if 'ip-adapter' in x:
            xlinkthis(f"/kaggle/input/sdxl_controlnets/pytorch/default/3/{x}", f"/kaggle/working/ComfyUI/models/ipadapter/{x}")

    # Link models from /kaggle/input/pulid-model/pytorch/default/3/
    for x in os.listdir("/kaggle/input/pulid-model/pytorch/default/3/"):
        print(x)
        xlinkthis(f"/kaggle/input/pulid-model/pytorch/default/3/{x}", f"/kaggle/working/ComfyUI/models/pulid/{x}")

    print("="*60, "Additional models linked successfully.", "="*60, sep="\n")

def start_comfyui_instances():
    """Start ComfyUI instances."""
    print("="*60, "Starting ComfyUI instances...", "-"*60, sep="\n")
    os.chdir("/kaggle/working/ComfyUI/")
    xP1 = subprocess.Popen(["python", "main.py", "--cuda-device", "0", "--port", "8188", "--highvram"])
    time.sleep(10)
    xP2 = subprocess.Popen(["python", "main.py", "--cuda-device", "1", "--port", "8189", "--highvram"])
    time.sleep(10)
    print("="*60, "ComfyUI instances started successfully.", "-"*60, sep="\n")

def start_playit_agent():
    """Start the Playit agent."""
    print("="*60, "Starting Playit agent...", "-"*60, sep="\n")
    os.chdir("/kaggle/working/ComfyUI/")
    if not os.path.exists('/kaggle/working/ComfyUI/playit-linux-amd64'):
        get_ipython().system('wget https://github.com/playit-cloud/playit-agent/releases/download/v0.15.26/playit-linux-amd64')
    get_ipython().system('chmod +x ./playit-linux-amd64')
    os.makedirs("/kaggle/working/ComfyUI/logs", exist_ok=True)
    subprocess.Popen("nohup /kaggle/working/ComfyUI/playit-linux-amd64 >> /kaggle/working/ComfyUI/logs/playit.log 2>&1 &", bufsize=0, shell=True)
    time.sleep(4)
    get_ipython().system('tail -n 20 /kaggle/working/ComfyUI/logs/playit.log')
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

def main():
    """Main function to orchestrate the setup and execution."""
    print("="*60, "Starting main function...", "-"*60, sep="\n")
    # Install uv
    print("="*60, "Installing uv...", "-"*60, sep="\n")
    get_ipython().system('python -m pip install -U pip uv -q')

    # Example usage of zip_folder
    folder_to_zip = '/kaggle/working/ComfyUI/output'
    if os.path.exists(folder_to_zip):
        output_zip = f'/kaggle/working/output_{datetime.now():%d%m%Y_%H%M}.zip'
        zip_folder(folder_to_zip, output_zip)
    
        # Move files
        x_old = '/kaggle/working/old_output'
        move_files(folder_to_zip, x_old)

    # Setup ComfyUI
    setup_comfyui()
    
    # Install packages
    packages = [
        "https://github.com/comfyanonymous/ComfyUI_bitsandbytes_NF4",
        "https://github.com/WASasquatch/was-node-suite-comfyui",
        "https://github.com/rgthree/rgthree-comfy",
        "https://codeberg.org/Gourieff/comfyui-reactor-node",
        "https://github.com/BlenderNeko/ComfyUI_Noise",
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
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
        "https://github.com/revirevy/Comfyui_saveimage_imgbb",
        "https://github.com/ltdrdata/ComfyUI-Manager",
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
        "https://github.com/godmt/ComfyUI-List-Utils"
    ]
    for package in packages:
        install_package(package)

    # Setup ComfyUI
    setup_comfyui()

    # Link models
    link_models()
    
    # link_additional_models
    link_additional_models()
    
    # Start ComfyUI instances
    start_comfyui_instances()

    # Start Playit agent
    start_playit_agent()

    # Start scheduler
    scheduler_thread = threading.Thread(target=scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

    print("="*60, "Scheduler started in background. It will execute the first cell every 1 minute.", "-"*60, sep="\n")

if __name__ == "__main__":
    main()
