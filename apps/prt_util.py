import os
import shutil
import math
import argparse
import trimesh
import numpy as np
from scipy.special import sph_harm
from tqdm import tqdm


def factratio(N, D):
    if N >= D:
        prod = 1.0
        for i in range(D + 1, N + 1):
            prod *= i
        return prod
    else:
        prod = 1.0
        for i in range(N + 1, D + 1):
            prod *= i
        return 1.0 / prod


def KVal(M, L):
    return math.sqrt(((2 * L + 1) / (4 * math.pi)) * (factratio(L - M, L + M)))


def AssociatedLegendre(M, L, x):
    if M < 0 or M > L or np.max(np.abs(x)) > 1.0:
        return np.zeros_like(x)

    pmm = np.ones_like(x)
    if M > 0:
        somx2 = np.sqrt((1.0 + x) * (1.0 - x))
        fact = 1.0
        for i in range(1, M + 1):
            pmm = -pmm * fact * somx2
            fact = fact + 2

    if L == M:
        return pmm
    else:
        pmmp1 = x * (2 * M + 1) * pmm
        if L == M + 1:
            return pmmp1
        else:
            pll = np.zeros_like(x)
            for i in range(M + 2, L + 1):
                pll = (x * (2 * i - 1) * pmmp1 - (i + M - 1) * pmm) / (i - M)
                pmm = pmmp1
                pmmp1 = pll
            return pll


def SphericalHarmonic(M, L, theta, phi):
    if M > 0:
        return (
            math.sqrt(2.0)
            * KVal(M, L)
            * np.cos(M * phi)
            * AssociatedLegendre(M, L, np.cos(theta))
        )
    elif M < 0:
        return (
            math.sqrt(2.0)
            * KVal(-M, L)
            * np.sin(-M * phi)
            * AssociatedLegendre(-M, L, np.cos(theta))
        )
    else:
        return KVal(0, L) * AssociatedLegendre(0, L, np.cos(theta))


def save_obj(mesh_path, verts):
    file = open(mesh_path, "w")
    for v in verts:
        file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    file.close()


def sampleSphericalDirections(n):
    xv = np.random.rand(n, n)
    yv = np.random.rand(n, n)
    theta = np.arccos(1 - 2 * xv)
    phi = 2.0 * math.pi * yv

    phi = phi.reshape(-1)
    theta = theta.reshape(-1)

    vx = -np.sin(theta) * np.cos(phi)
    vy = -np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)
    return np.stack([vx, vy, vz], 1), phi, theta


def getSHCoeffs(order, phi, theta):
    shs = []
    for n in range(0, order + 1):
        for m in range(-n, n + 1):
            s = SphericalHarmonic(m, n, theta, phi)
            shs.append(s)

    return np.stack(shs, 1)


def computePRT(mesh_path, n, order, type="rp"):

    vectors_orig, phi, theta = sampleSphericalDirections(n)
    SH_orig = getSHCoeffs(order, phi, theta)

    w = 4.0 * math.pi / (n * n)

    if type == "nba":
        mesh = trimesh.load(
            mesh_path,
            process=False,
            force="mesh",
            skip_materials=True,
            # maintain_order=True
        )
    else:
        mesh = trimesh.load(
                mesh_path,
                process=False,
            )

    origins = mesh.vertices
    normals = mesh.vertex_normals
    n_v = origins.shape[0]

    origins = np.repeat(origins[:, None], n, axis=1).reshape(-1, 3)
    normals = np.repeat(normals[:, None], n, axis=1).reshape(-1, 3)
    PRT_all = None
    for i in tqdm(range(n)):
        SH = np.repeat(SH_orig[None, (i * n): ((i + 1) * n)], n_v, axis=0).reshape(
            -1, SH_orig.shape[1]
        )
        vectors = np.repeat(
            vectors_orig[None, (i * n): ((i + 1) * n)], n_v, axis=0
        ).reshape(-1, 3)

        dots = (vectors * normals).sum(1)
        front = dots > 0.0

        delta = 1e-3 * min(mesh.bounding_box.extents)
        hits = mesh.ray.intersects_any(origins + delta * normals, vectors)
        nohits = np.logical_and(front, np.logical_not(hits))

        PRT = (nohits.astype(np.float) * dots)[:, None] * SH

        if PRT_all is not None:
            PRT_all += PRT.reshape(-1, n, SH.shape[1]).sum(1)
        else:
            PRT_all = PRT.reshape(-1, n, SH.shape[1]).sum(1)

    PRT = w * PRT_all

    # NOTE: trimesh sometimes break the original vertex order, but topology will not change.
    # when loading PRT in other program, use the triangle list from trimesh.
    return PRT, mesh.faces


"""
-renderpeople dataset
    directory: rp_dennis_posed/004_OBJ

-twindom dataset
    directory: M1, M2, etc
    - contains .mtl, .obj, .png
    https://web.twindom.com/human-3d-body-scan-dataset-for-research-from-3d-scans/

-nba2k dataset
    e.g.
    cedric
    |
    |____ 2ku --> frame_count --> players --> .obj + .obj.mtl (single player)
    |
    |____ normal --> frame_count --> players --> .obj + .obj.mtl (multiple players)
    |
    |____ rest_pose --> single frame_count --> .obj + .obj.mtl (single player)
    |
    |____ texture --> .png files for textures (e.g. arm, eye, hair, hair...)
    |
    |____ resampled --> not used in our case since it doesn't have hairs etc
"""


def testPRT(dir_path, type, n=40):
    if dir_path[-1] == "/":
        dir_path = dir_path[:-1]
    if type == "rp":
        sub_name = dir_path.split("/")[-1][:-4]
        obj_path = os.path.join(dir_path, sub_name + "_100k.obj")
    elif type == "twindom":
        for file in os.listdir(dir_path):
            if ".obj" in file:
                obj_file = file
                prefix = obj_file.split(".obj")[0]
                break
        for file in os.listdir(dir_path):
            # twindom dataset doesn't have matching png file sometimes
            if prefix not in file:
                raise ValueError(f"Invalid prefix type for file {file}")
        obj_path = os.path.join(dir_path, obj_file)
    elif type == "nba":
        # copy textures to 2ku or rest_pose directory since mtl points to wrong directory now
        # e.g. data/nba_dataset/mesh/release/cedric/rest_pose/NBA2K19_2020.05.27_21.05.46_frame49630/players
        if "2ku" in dir_path or "rest_pose" in dir_path:
            obj_path = os.path.join(dir_path, "0_person.obj")

            # original texture directory (src)
            textures_dir = os.path.join(
                "/".join(dir_path.split("/")[:-3]), "textures"
            )
            # frame texture directory (dest)
            frame_texture_dir = os.path.join(
                "/".join(dir_path.split("/")[:-2]), "textures"
            )

            # copy the file over if doesn't exist
            if not os.makedirs(frame_texture_dir, exist_ok=True):
                for item in os.listdir(textures_dir):
                    shutil.copy2(
                        os.path.join(textures_dir, item),
                        os.path.join(frame_texture_dir, item)
                    )
        else:
            # need to handle the case where some have multiple players
            raise TypeError("normal dataset for NBA is not supported yet.")

    os.makedirs(os.path.join(dir_path, "bounce"), exist_ok=True)

    PRT, F = computePRT(obj_path, n, 2, type)
    np.savetxt(os.path.join(dir_path, "bounce",
                            "bounce0.txt"), PRT, fmt="%.8f")
    np.save(os.path.join(dir_path, "bounce", "face.npy"), F)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="/home/shunsuke/Downloads/rp_dennis_posed_004_OBJ",
    )
    parser.add_argument(
        "-n",
        "--n_sample",
        type=int,
        default=40,
        help="squared root of number of sampling. the higher, the more accurate, but slower",
    )
    parser.add_argument("-t", "--type", type=str,
                        default="rp", help="type of dataset")
    args = parser.parse_args()

    testPRT(args.input, args.type)
