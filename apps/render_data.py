#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path
import os
import cv2
import time
import math
import random
import pyexr
import argparse
import shutil
from tqdm import tqdm

from lib.renderer.camera import Camera
import numpy as np
import trimesh
from lib.renderer.mesh import (
    load_obj_mesh,
    compute_tangent,
    compute_normal,
    load_obj_mesh_mtl,
)
from lib.renderer.camera import Camera

SCALE_MAPPING = {
    "head.png": 0.75,
    "arm.png": 0.75,
    "shoe.png": 0.375,
    "shirt.png": 0.75,
    "eye.png": 0.09375,
    "tooth.png": 0.1875,
    "pant.png": 0.375,
    "hair.png": 0.375,
    "mane.png": 0.1875,
    "leg.png": 0.375,
    "beard.png": 0.375
}

TRANSLATION_MAPPING = {
    "head.png": [[-0.38, 0.00]], # head
    "arm.png": [[-1.0, -1.0]], # arm
    "shoe.png_1": [[0.5, 0.5]], # shoe stripe 1
    "shoe.png_2": [[0.48, -0.225]], # shoe
    "shoe.png_3": [[0.0, 0.3]], # shoe stripe 2
    "shoe.png_4": [[0.0, -1.0]], # shoe
    "shirt.png": [[0.2, -1.05]], # shirt
    "eye.png": [[-0.95, 0.3]], # eye
    "tooth.png": [[-0.95, -0.2]], # tooth
    "pant.png": [[-0.95, 0.4]], # pant
    "hair.png": [[0.5, -0.205]], # hair
    "mane.png": [[-0.25, -0.48]], # mane
    "leg.png": [[-0.25, -0.95]], # leg
    "beard.png": [[-0.71, -0.18]] # beard
}

def center_normalize_mesh(mesh):
    extents = mesh.extents
    vmin = mesh.vertices.min(0)
    vmax = mesh.vertices.max(0)
    center = 0.5 * (vmin+vmax)

    m = trimesh.transformations.translation_matrix(-center)
    meshPre = mesh.apply_transform(m)
    m = trimesh.transformations.scale_matrix(1/np.max(extents))
    mesh = meshPre.apply_transform(m)

    return mesh

def make_rotate(rx, ry, rz):

    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


def rotateSH(SH, R):
    SHn = SH

    # 1st order
    SHn[1] = R[1, 1] * SH[1] - R[1, 2] * SH[2] + R[1, 0] * SH[3]
    SHn[2] = -R[2, 1] * SH[1] + R[2, 2] * SH[2] - R[2, 0] * SH[3]
    SHn[3] = R[0, 1] * SH[1] - R[0, 2] * SH[2] + R[0, 0] * SH[3]

    # 2nd order
    SHn[4:, 0] = rotateBand2(SH[4:, 0], R)
    SHn[4:, 1] = rotateBand2(SH[4:, 1], R)
    SHn[4:, 2] = rotateBand2(SH[4:, 2], R)

    return SHn


def rotateBand2(x, R):
    s_c3 = 0.94617469575
    s_c4 = -0.31539156525
    s_c5 = 0.54627421529

    s_c_scale = 1.0 / 0.91529123286551084
    s_c_scale_inv = 0.91529123286551084

    s_rc2 = 1.5853309190550713 * s_c_scale
    s_c4_div_c3 = s_c4 / s_c3
    s_c4_div_c3_x2 = (s_c4 / s_c3) * 2.0

    s_scale_dst2 = s_c3 * s_c_scale_inv
    s_scale_dst4 = s_c5 * s_c_scale_inv

    sh0 = x[3] + x[4] + x[4] - x[1]
    sh1 = x[0] + s_rc2 * x[2] + x[3] + x[4]
    sh2 = x[0]
    sh3 = -x[3]
    sh4 = -x[1]

    r2x = R[0][0] + R[0][1]
    r2y = R[1][0] + R[1][1]
    r2z = R[2][0] + R[2][1]

    r3x = R[0][0] + R[0][2]
    r3y = R[1][0] + R[1][2]
    r3z = R[2][0] + R[2][2]

    r4x = R[0][1] + R[0][2]
    r4y = R[1][1] + R[1][2]
    r4z = R[2][1] + R[2][2]

    sh0_x = sh0 * R[0][0]
    sh0_y = sh0 * R[1][0]
    d0 = sh0_x * R[1][0]
    d1 = sh0_y * R[2][0]
    d2 = sh0 * (R[2][0] * R[2][0] + s_c4_div_c3)
    d3 = sh0_x * R[2][0]
    d4 = sh0_x * R[0][0] - sh0_y * R[1][0]

    sh1_x = sh1 * R[0][2]
    sh1_y = sh1 * R[1][2]
    d0 += sh1_x * R[1][2]
    d1 += sh1_y * R[2][2]
    d2 += sh1 * (R[2][2] * R[2][2] + s_c4_div_c3)
    d3 += sh1_x * R[2][2]
    d4 += sh1_x * R[0][2] - sh1_y * R[1][2]

    sh2_x = sh2 * r2x
    sh2_y = sh2 * r2y
    d0 += sh2_x * r2y
    d1 += sh2_y * r2z
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
    d3 += sh2_x * r2z
    d4 += sh2_x * r2x - sh2_y * r2y

    sh3_x = sh3 * r3x
    sh3_y = sh3 * r3y
    d0 += sh3_x * r3y
    d1 += sh3_y * r3z
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
    d3 += sh3_x * r3z
    d4 += sh3_x * r3x - sh3_y * r3y

    sh4_x = sh4 * r4x
    sh4_y = sh4 * r4y
    d0 += sh4_x * r4y
    d1 += sh4_y * r4z
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
    d3 += sh4_x * r4z
    d4 += sh4_x * r4x - sh4_y * r4y

    dst = x
    dst[0] = d0
    dst[1] = -d1
    dst[2] = d2 * s_scale_dst2
    dst[3] = -d3
    dst[4] = d4 * s_scale_dst4

    return dst


def render_prt_ortho(
    out_path,
    folder_name,
    type,
    subject_name,
    normalized,
    shs,
    rndr,
    rndr_uv,
    im_size,
    angl_step=4,
    n_light=1,
    pitch=[0],
):
    cam = Camera(width=im_size, height=im_size)
    cam.ortho_ratio = 0.4 * (512 / im_size)
    cam.near = -100
    cam.far = 100
    cam.sanity_check()

    # set path for obj, prt
    if type == "rp":
        mesh_file = os.path.join(folder_name, subject_name + "_100k.obj")
        text_file = os.path.join(
            folder_name, "tex", subject_name + "_dif_2k.jpg")
    elif type == "twindom":
        mesh_file = os.path.join(folder_name, subject_name + ".obj")
        for file in os.listdir(folder_name):
            if ".png" in file or ".jpg" in file:
                text_name = file
                break
        text_file = os.path.join(folder_name, text_name)
    elif type == "nba":
        # all files has 0_person.obj
        mesh_file = os.path.join(folder_name, "0_person.obj")
        # temporary text file path
        text_file = os.path.join(folder_name, "..", "..", "textures/shirt.png")

    if not os.path.exists(mesh_file):
        print("ERROR: obj file does not exist!!", mesh_file)
        return
    prt_file = os.path.join(folder_name, "bounce", "bounce0.txt")
    if not os.path.exists(prt_file):
        print("ERROR: prt file does not exist!!!", prt_file)
        return
    face_prt_file = os.path.join(folder_name, "bounce", "face.npy")
    if not os.path.exists(face_prt_file):
        print("ERROR: face prt file does not exist!!!", prt_file)
        return

    if not os.path.exists(text_file):
        print("ERROR: dif file does not exist!!", text_file)
        return

    texture_image = cv2.imread(text_file)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    # nba dataset load from mtl file
    if type == "nba":
        (
            vertices,
            faces,
            normals,
            faces_normals,
            textures,
            face_textures,
            face_data_mat,
            face_norm_data_mat,
            face_textures_data_mat,
            mtl_data
        ) = load_obj_mesh_mtl(mesh_file)
    else:
        (
            vertices,
            faces,
            normals,
            faces_normals,
            textures,
            face_textures,
        ) = load_obj_mesh(mesh_file, with_normal=True, with_texture=True)

    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1 if (vmax - vmin).argmax() == 1 else 2

    # Calculate median and scale for normalization
    vmed = np.median(vertices, 0)
    vmed[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
    y_scale = 180 / (vmax[up_axis] - vmin[up_axis])

    # set the scaling for the subject
    rndr.set_norm_mat(y_scale, vmed)
    rndr_uv.set_norm_mat(y_scale, vmed)

    tan, bitan = compute_tangent(
        vertices, faces, normals, textures, face_textures
    )
    prt = np.loadtxt(prt_file)
    # Loads the triangles meshes from trimesh since order is not preserved
    face_prt = np.load(face_prt_file)

    # scale and translation for nba
    scale_data = {}
    translation_data = {}

    if type == "nba":
        rndr.set_mesh_mtl(
            vertices=vertices,
            faces=face_data_mat,
            norms=normals,
            faces_nml=face_norm_data_mat,
            uvs=textures,
            faces_uvs=face_textures_data_mat,
            tans=tan,
            bitans=bitan,
            prt=prt,
            face_prt=face_prt
        )
        # set texture and their name
        for key in mtl_data:
            text_file = os.path.join(folder_name, mtl_data[key]['map_Kd'])
            texture_image = cv2.imread(text_file)
            texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
            rndr.set_albedo(
                texture_image=texture_image,
                mat_name=key
            )
            cnt = 1
            # e.g. '../../textures/head.png'
            # there's multiple shoes in nba dataset
            material_name = mtl_data[key]['map_Kd'].split("/")[-1]
            if 'shoe' not in mtl_data[key]['map_Kd']:
                translation_data[key] = TRANSLATION_MAPPING[material_name]
            else:
                translation_data[key] = TRANSLATION_MAPPING[material_name + f'_{cnt}']
                cnt += 1
            scale_data[key] = SCALE_MAPPING[material_name]
    else:
        rndr.set_mesh(
            vertices,
            faces,
            normals,
            faces_normals,
            textures,
            face_textures,
            prt,
            face_prt,
            tan,
            bitan,
        )
        rndr.set_albedo(texture_image)

    if type == "nba":
        rndr_uv.set_mesh_mtl(
            vertices=vertices,
            faces=face_data_mat,
            norms=normals,
            faces_nml=face_norm_data_mat,
            uvs=textures,
            faces_uvs=face_textures_data_mat,
            tans=tan,
            bitans=bitan,
            prt=prt,
            face_prt=face_prt,
            tex_offset=translation_data,
            scale=scale_data
        )
        # set texture and their name
        for key in mtl_data:
            text_file = os.path.join(folder_name, mtl_data[key]['map_Kd'])
            texture_image = cv2.imread(text_file)
            texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
            rndr_uv.set_albedo(
                texture_image=texture_image,
                mat_name=key
            )
    else:
        rndr_uv.set_mesh(
            vertices,
            faces,
            normals,
            faces_normals,
            textures,
            face_textures,
            prt,
            face_prt,
            tan,
            bitan,
        )
        rndr_uv.set_albedo(texture_image)

    os.makedirs(os.path.join(out_path, "GEO",
                             "OBJ", subject_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, "PARAM", subject_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, "RENDER", subject_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, "MASK", subject_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, "UV_RENDER",
                             subject_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, "UV_MASK", subject_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, "UV_POS", subject_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, "UV_NORMAL",
                             subject_name), exist_ok=True)

    if not os.path.exists(os.path.join(out_path, "val.txt")):
        f = open(os.path.join(out_path, "val.txt"), "w")
        f.close()

    if not normalized:
        # copy obj file
        cmd = "cp %s %s" % (mesh_file, os.path.join(
          out_path, "GEO", "OBJ", subject_name))
        os.system(cmd)
    else: # don't use this
        obj_name = mesh_file.split("/")[-1]
        mesh = trimesh.load(mesh_file, process=False, force="mesh", skip_materials=True)
        mesh = center_normalize_mesh(mesh)

        with open(os.path.join(out_path, "GEO", "OBJ", subject_name, obj_name), "w") as f:
            mesh.export(f, "obj")

    if type == "nba":
        # rename 0_person.obj to subject name
        src_path = os.path.join(
            out_path, "GEO", "OBJ", subject_name, "0_person.obj")
        dest_path = src_path.replace("0_person", subject_name)
        shutil.copyfile(src_path, dest_path)

    for p in pitch:
        for y in tqdm(range(0, 360, angl_step)):
            # rotation matrix for the subject
            R = np.matmul(
                make_rotate(math.radians(p), 0, 0), make_rotate(
                    0, math.radians(y), 0)
            )
            if up_axis == 2:
                R = np.matmul(R, make_rotate(math.radians(90), 0, 0))

            rndr.rot_matrix = R
            rndr_uv.rot_matrix = R
            rndr.set_camera(cam)
            rndr_uv.set_camera(cam)

            for j in range(n_light):
                sh_id = random.randint(0, shs.shape[0] - 1)
                sh = shs[sh_id]
                sh_angle = 0.2 * np.pi * (random.random() - 0.5)
                sh = rotateSH(sh, make_rotate(0, sh_angle, 0).T)

                dic = {
                    "sh": sh,
                    "ortho_ratio": cam.ortho_ratio,
                    "scale": y_scale,
                    "center": vmed,
                    "R": R,
                }

                rndr.set_sh(sh)
                rndr.analytic = True
                rndr.use_inverse_depth = False
                rndr.display()

                out_all_f = rndr.get_color(0)
                out_mask = out_all_f[:, :, 3]
                out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGR)

                np.save(
                    os.path.join(
                        out_path, "PARAM", subject_name, "%d_%d_%02d.npy" % (
                            y, p, j)
                    ),
                    dic,
                )
                cv2.imwrite(
                    os.path.join(
                        out_path, "RENDER", subject_name, "%d_%d_%02d.jpg" % (
                            y, p, j)
                    ),
                    255.0 * out_all_f,
                )
                cv2.imwrite(
                    os.path.join(
                        out_path, "MASK", subject_name, "%d_%d_%02d.png" % (
                            y, p, j)
                    ),
                    255.0 * out_mask,
                )

                # Draw the UV mapping here
                rndr_uv.set_sh(sh)
                rndr_uv.analytic = True
                rndr_uv.use_inverse_depth = False
                rndr_uv.display()

                uv_color = rndr_uv.get_color(0)
                uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(
                    os.path.join(
                        out_path,
                        "UV_RENDER",
                        subject_name,
                        "%d_%d_%02d.jpg" % (y, p, j),
                    ),
                    255.0 * uv_color,
                )

                if y == 0 and j == 0 and p == pitch[0]:
                    uv_pos = rndr_uv.get_color(1)
                    uv_mask = uv_pos[:, :, 3]
                    cv2.imwrite(
                        os.path.join(out_path, "UV_MASK",
                                     subject_name, "00.png"),
                        255.0 * uv_mask,
                    )

                    # default is a reserved name
                    data = {"default": uv_pos[:, :, :3]}
                    pyexr.write(
                        os.path.join(out_path, "UV_POS",
                                     subject_name, "00.exr"), data
                    )

                    uv_nml = rndr_uv.get_color(2)
                    uv_nml = cv2.cvtColor(uv_nml, cv2.COLOR_RGBA2BGR)
                    cv2.imwrite(
                        os.path.join(out_path, "UV_NORMAL",
                                     subject_name, "00.png"),
                        255.0 * uv_nml,
                    )


if __name__ == "__main__":
    shs = np.load("./env_sh.npy")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        default="/home/shunsuke/Downloads/rp_dennis_posed_004_OBJ")
    parser.add_argument(
        "-o", "--out_dir", type=str, default="/home/shunsuke/Documents/hf_human"
    )
    parser.add_argument(
        "-n", "--normalized", action="store_true"
    )
    parser.add_argument(
        "-m",
        "--ms_rate",
        type=int,
        default=1,
        help="higher ms rate results in less aliased output. MESA renderer only supports ms_rate=1.",
    )
    parser.add_argument(
        "-e",
        "--egl",
        action="store_true",
        help="egl rendering option. use this when rendering with headless server with NVIDIA GPU",
    )
    parser.add_argument(
        "-s", "--size", type=int, default=512, help="rendering image size"
    )
    parser.add_argument("-t", "--type", type=str,
                        default="rp", help="type of dataset")
    args = parser.parse_args()

    # NOTE: GL context has to be created before any other OpenGL function loads.
    from lib.renderer.gl.init_gl import initialize_GL_context

    initialize_GL_context(width=args.size, height=args.size, egl=args.egl)

    if args.input[-1] == "/":
        args.input = args.input[:-1]

    """
        Check for dataset type
    """
    if args.type == "rp":
        subject_name = args.input.split("/")[-1][:-4]
    elif args.type == "twindom":
        for file in os.listdir(args.input):
            if ".obj" in file:
                obj_file = file
                subject_name = obj_file.split(".obj")[0]
                break
    elif args.type == "nba":
        input = args.input
        player_name = input.split("/")[-4]
        frame_type = input.split("/")[-3]
        frame = input.split("/")[-2]
        subject_name = player_name + "_" + frame_type + "_" + frame

    from lib.renderer.gl.prt_render import PRTRender

    rndr = PRTRender(
        width=args.size, height=args.size, ms_rate=args.ms_rate, egl=args.egl, type=args.type
    )
    rndr_uv = PRTRender(
        width=args.size, height=args.size, uv_mode=True, egl=args.egl, type=args.type
    )

    render_prt_ortho(
        args.out_dir,
        args.input,
        args.type,
        subject_name,
        args.normalized,
        shs,
        rndr,
        rndr_uv,
        args.size,
        1,
        1,
        pitch=[0],
    )
