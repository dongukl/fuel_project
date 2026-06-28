#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_aruco_marker_grid_visual_test.py

Isaac Sim Script Editor에서 실행한다 (텍스처 파일 불필요, Cube geometry로 생성).

배경
- world.usda의 /World/aruco_vehicle_marker가 카메라 기준 좌우 반전(mirror)되어 보여서
  aruco_marker_detector.py가 마커를 검출하지 못했다.
- ArUco 디코더는 회전(0/90/180/270)은 보정하지만 좌우 반전은 보정하지 않으므로,
  거울상으로 보이는 패턴은 DICT_4X4_50의 어떤 유효 id와도 맞지 않아 그냥 버려진다.
- 카메라가 보는 면을 바꿀 수 없으므로, "거울상의 거울상 = 원본"이 되도록
  DICT_4X4_50 id=0의 원본 패턴을 열(column) 기준으로 미리 반전해서 굽는다(BAKED_PATTERN).

이 스크립트는 기존 /World/aruco_vehicle_marker를 삭제하고 같은 위치/크기로 재생성한다.
위치를 옮기는 게 아니라 "찍히는 패턴"만 고치는 것이므로, 차량에 붙인 물리적 위치는 그대로 유지된다.
"""
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade

MARKER_PRIM_PATH = "/World/aruco_vehicle_marker"

# world.usda에 기존에 배치되어 있던 값과 동일하게 맞춘다 (차량 부착 위치 유지).
MARKER_CENTER_X = 0.468191
MARKER_CENTER_Z = 0.85052
BACKER_Y = -1.252334   # white backer plate (차체 표면)
CELL_Y = -1.250834     # black cell은 backer보다 1.5mm 카메라 쪽으로 띄워서 z-fighting 방지

CELL_SIZE = 0.02       # 6x6 grid -> marker_size_m = 6 * 0.02 = 0.12 (aruco_detector_params_visual_test.yaml과 일치)
BACKER_SIZE = 0.15      # quiet zone(여백) 포함 backer 크기

# DICT_4X4_50, id=0의 "원본" 6x6 비트 패턴 (1=black, 0=white). row0=위쪽 테두리, row5=아래쪽 테두리.
# cv2.aruco.generateImageMarker(getPredefinedDictionary(DICT_4X4_50), 0, ...) 로 검증한 값.
ORIGINAL_PATTERN = [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
]

# 카메라가 마커의 반대쪽 면을 보고 있어 좌우가 뒤집혀 보이므로, 열 순서를 미리 반전해서 굽는다.
BAKED_PATTERN = [row[::-1] for row in ORIGINAL_PATTERN]


def _delete_existing(stage) -> None:
    prim = stage.GetPrimAtPath(MARKER_PRIM_PATH)
    if prim.IsValid():
        stage.RemovePrim(MARKER_PRIM_PATH)


def _make_material(stage, path: str, color: tuple) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput("surface").ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _make_cube(stage, path: str, translate: tuple, scale: tuple, material: UsdShade.Material) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))
    UsdShade.MaterialBindingAPI(cube).Bind(material)


def main() -> None:
    stage = omni.usd.get_context().get_stage()

    _delete_existing(stage)
    UsdGeom.Xform.Define(stage, MARKER_PRIM_PATH)

    looks_path = MARKER_PRIM_PATH + "/Looks"
    black_mat = _make_material(stage, looks_path + "/black", (0.02, 0.02, 0.02))
    white_mat = _make_material(stage, looks_path + "/white", (0.9, 0.9, 0.9))

    _make_cube(
        stage, MARKER_PRIM_PATH + "/white_backer",
        translate=(MARKER_CENTER_X, BACKER_Y, MARKER_CENTER_Z),
        scale=(BACKER_SIZE, 0.001, BACKER_SIZE),
        material=white_mat,
    )

    half = (len(BAKED_PATTERN) - 1) / 2.0  # 2.5
    for r, row in enumerate(BAKED_PATTERN):
        for c, bit in enumerate(row):
            if not bit:
                continue  # white는 backer 색이 그대로 보이므로 cube를 만들지 않는다.
            x = MARKER_CENTER_X + (c - half) * CELL_SIZE
            z = MARKER_CENTER_Z - (r - half) * CELL_SIZE  # row0이 +z(위쪽)
            _make_cube(
                stage, f"{MARKER_PRIM_PATH}/black_cell_{r}_{c}",
                translate=(x, CELL_Y, z),
                scale=(CELL_SIZE, 0.001, CELL_SIZE),
                material=black_mat,
            )

    print(f"[aruco] {MARKER_PRIM_PATH} 재생성 완료 (좌우 반전 보정 적용, DICT_4X4_50 id=0)")


main()
