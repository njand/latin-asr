import os
import shutil

import modal

app = modal.App("wipe-latin-cache")
cache_volume = modal.Volume.from_name("latin-audio-cache")

@app.function(volumes={"/mnt/cache": cache_volume})
def wipe():
    if os.path.exists("/mnt/cache"):
        for item in os.listdir("/mnt/cache"):
            item_path = os.path.join("/mnt/cache", item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print("Successfully wiped /mnt/cache!")
    cache_volume.commit()

@app.local_entrypoint()
def main():
    wipe.remote()