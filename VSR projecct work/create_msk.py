root_db = "./"
import os
import glob
import tqdm
import json
import numpy as np
from PIL import Image
from base64 import b64decode
import zlib
from io import BytesIO

def decode(s):
    s = b64decode(s)
    try:
        s = zlib.decompress(s)
    except zlib.error:
        # If the string is not compressed, we'll not use zlib.
        pass
    with BytesIO(s) as d:
        img = Image.open(d)
        img.load()
    return img

list_json = glob.glob(os.path.join(root_db, 't*/ann/*.json'))
for josn_file in tqdm.tqdm(list_json):
    img_file = os.path.join(
        os.path.dirname(os.path.dirname(josn_file)),
        'img', os.path.basename(josn_file)[:-5])
    msk_file = os.path.join(
        os.path.dirname(os.path.dirname(josn_file)),
        'msk', os.path.basename(josn_file)[:-5]+'.png')
      
    with open(josn_file) as fid:
      dat = json.load(fid)
    height = dat["size"]["height"]
    width = dat["size"]["width"]
    msk = np.zeros((height,width), np.uint16)
    for obj in dat["objects"]:
        assert obj["geometryType"] == "bitmap"
        assert obj["classTitle"] == "blood_cell"
        map = decode(obj["bitmap"]["data"])
        org = obj["bitmap"]['origin']
        assert map.mode == 'P'
        assert len(map.getpalette())==6
        map = np.asarray(map)
        msk[org[1]:org[1]+map.shape[0],
            org[0]:org[0]+map.shape[1]] = map
    os.makedirs(os.path.dirname(msk_file), exist_ok=True)
    msk = Image.fromarray(np.uint8(msk.clip(0,1)), mode='P')
    msk.putpalette([0, 0, 0, 255, 255, 255])
    msk.save(msk_file)
