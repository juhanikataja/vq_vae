import pytools as pt
import numpy as np 
import mmap
import torch
import os

#Reads in a VDF from cid CellID in a 3D  32 bit numpy array
def extract_vdf(file, cid, box=-1):
    assert cid > 0
    f = pt.vlsvfile.VlsvReader(file)
    # -- read phase space density
    vcells = f.read_velocity_cells(cid)
    keys = list(vcells.keys())
    values = list(vcells.values())

    # -- generate a velocity space
    size = f.get_velocity_mesh_size()
    vids = np.arange(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))

    # -- put phase space density into array
    dist = np.zeros_like(vids, dtype=float)
    dist.fill(np.NaN)
    dist.fill(0)
    dist[keys] = values

    # -- sort vspace by velocity
    v = f.get_velocity_cell_coordinates(vids)

    i = np.argsort(v[:, 0], kind="stable")
    v = v[i]
    # vids = vids[i]
    dist = dist[i]

    j = np.argsort(v[:, 1], kind="stable")
    v = v[j]
    # vids = vids[j]
    dist = dist[j]

    k = np.argsort(v[:, 2], kind="stable")
    v = v[k]
    # vids = vids[k]
    dist = dist[k]
    dist = dist.reshape(4 * int(size[0]), 4 * int(size[1]), 4 * int(size[2]))
    vdf = dist
    i, j, k = np.unravel_index(np.nanargmax(vdf), vdf.shape)
    len = box
    data = vdf[(i - len) : (i + len), (j - len) : (j + len), (k - len) : (k + len)]
    return np.array(data, dtype=np.float32)


def extract_vdf_reader(f, cid, box=-1):
    assert cid > 0
    # -- read phase space density
    vcells = f.read_velocity_cells(cid)
    keys = list(vcells.keys())
    values = list(vcells.values())

    # -- generate a velocity space
    size = f.get_velocity_mesh_size()
    vids = np.arange(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))

    # -- put phase space density into array
    dist = np.zeros_like(vids, dtype=float)
    dist.fill(np.NaN)
    dist.fill(0)
    dist[keys] = values

    # -- sort vspace by velocity
    v = f.get_velocity_cell_coordinates(vids)

    i = np.argsort(v[:, 0], kind="stable")
    v = v[i]
    # vids = vids[i]
    dist = dist[i]

    j = np.argsort(v[:, 1], kind="stable")
    v = v[j]
    # vids = vids[j]
    dist = dist[j]

    k = np.argsort(v[:, 2], kind="stable")
    v = v[k]
    # vids = vids[k]
    dist = dist[k]
    dist = dist.reshape(4 * int(size[0]), 4 * int(size[1]), 4 * int(size[2]))
    vdf = dist
    i, j, k = np.unravel_index(np.nanargmax(vdf), vdf.shape)
    len = box
    data = vdf[(i - len) : (i + len), (j - len) : (j + len), (k - len) : (k + len)]
    return np.array(data, dtype=np.float32)


def extract_vdfs(file , cids,box):
    vdfs=[]
    for cid in cids:
        vdfs.append(extract_vdf(file, cid,box))
    return np.stack(vdfs);

#Hackity hack do not know hwo else to do this in python
def create_memory_mapped_file(filename, bytes):
    with open(filename, "wb+") as f:
        f.seek(bytes - 1)
        f.write(b'\x00')
        f.flush()
        return mmap.mmap(-1, bytes, prot=mmap.PROT_READ)

#Returns a memory maped  anonymous file which contains all the vdfs
def create_restart_mapping(filename,cids,box):
    bytes_per_vdf=2*box*2*box*2*box*4 #(box is half-width and **4** for 4-byte float32)
    total_size_of_mapping_bytes=bytes_per_vdf*len(cids) 
    print(f"Creating mapped restart region with expected size {total_size_of_mapping_bytes} bytes")
    f = pt.vlsvfile.VlsvReader(filename)
    mmapped=mmap.mmap(-1, total_size_of_mapping_bytes)
    mmapped.seek(0)
    for (i,cid) in enumerate(cids):
        vdf=extract_vdf_reader(f, cid,box)
        print(f"Mapping {i}th VDF...")
        index_in_mapping=i*bytes_per_vdf
        #Here we seek instead of [] to just perform some error checking
        mmapped.seek(index_in_mapping)
        sz=mmapped.write(vdf.tobytes())
        assert sz==bytes_per_vdf
    return mmapped,bytes_per_vdf

class Vlasiator_DataSet():
    def __init__(self, cids,filename,device,box=25):
        self.cids=cids
        self.box=box
        self.device=device
        self.f=pt.vlsvfile.VlsvReader(filename)

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        vdf=extract_vdf_reader(self.f, self.cids[idx],self.box)
        vdf_norm = (vdf - vdf.min())/(vdf.max() - vdf.min())
        return torch.tensor(vdf_norm).unsqueeze(0).to(self.device)
        

class MMapped_Vlasiator_DataSet():
    def __init__(self, cids,filename,device,box=25):
        self.cids=cids
        self.box=box
        self.device=device
        self.mmapped,self.bytes_per_vdf=create_restart_mapping(filename,cids,box);
    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        index_in_mapping=self.bytes_per_vdf*idx;
        vdf=np.frombuffer(self.mmapped[index_in_mapping:index_in_mapping+self.bytes_per_vdf],dtype=np.float32).reshape((2*self.box,2*self.box,2*self.box))
        vdf_norm = (vdf - vdf.min())/(vdf.max() - vdf.min())
        return torch.tensor(vdf_norm).unsqueeze(0).to(self.device)
        
