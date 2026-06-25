# visual_test.usda ArUco 기준값 요약

## 선택한 marker 설정
- Dictionary: DICT_4X4_50
- Marker ID: 0
- Marker pattern size: 0.12 m
- Marker world center: [0.468191, -1.252334, 0.85052]  (world.usda의 black_cell/white_backer cube 실측값, 2026-06-24 갱신, 2026-06-25 재확인 시에도 동일)
- Marker path: /World/aruco_vehicle_marker

## marker frame convention (2026-06-25 rqt 실측으로 정정)
- x = 좌우 (world X, 부호 반전)
- y = 깊이 (world Y, 부호 반전, 카메라가 있는 방향)
- z = 높이 (world Z, 부호 반전)

즉 퍼뮤테이션 없이 축별로 그대로 대응되고 부호만 반전된다:

```
marker_to_target_xyz = MARKER_CENTER_WORLD - target_world
```

(과거에는 x=-X, y=-Z, z=-Y로 y/z가 뒤섞인 퍼뮤테이션을 썼는데, rqt에서 실제로 y를 바꾸면 깊이가,
z를 바꾸면 높이가 움직이는 걸 확인하고 위 식으로 정정했다. 옛 하드코딩 FUEL_CAP_CENTER 등을 이 식으로
역산해보면 정확히 일치하므로, 과거 계산도 실제로는 이 식을 썼던 것으로 보인다 — 문서 설명만 잘못 적혀 있었음.)

create_aruco_marker_grid_visual_test.py의 BAKED_PATTERN(좌우반전 보정)과는 무관하다(그건 마커 패턴 자체의
좌우반전 보정이고, 이 frame convention은 검출된 marker pose의 offset 축 해석이다).

## world.usda 실제 좌표 (2026-06-25, payload 경로 버그 수정 후 재계산)

world.usda의 차량(`car_visual`) payload 참조 경로가 `../Desktop/rokey_F4/cobot3_ws/...`로 되어 있어
(이 파일이 이미 `rokey_F4` 안에 있으므로 `Desktop/Desktop/rokey_F4/...`로 잘못 풀리던) 버그를 `cobot3_ws/...`로
수정한 뒤 USD 라이브러리로 직접 world transform을 읽었다.

- fuel_door world: [0.240267, -1.279059, 1.110197]
- fuel_cap  world: [0.238191, -1.302334, 1.090520]
- fuel_port_hole world: [0.244428, -1.510171, 1.054349]

이전 multi_robot_oiling.py의 하드코딩 FUEL_DOOR_CENTER/FUEL_CAP_CENTER/FUEL_PORT_HOLE_CENTER 값과는
크게 다르다 — 이번 world.usda 업데이트로 차량이 마커 쪽으로 더 가깝게(다른 배치로) 옮겨졌기 때문이다.
(multi_robot_oiling.py의 하드코딩 fallback 값은 아직 갱신하지 않았다. USD에서 prim을 못 찾을 때만 쓰는
fallback이라 당장 동작에는 영향 없지만, 필요하면 같이 갱신할 것.)

## target offsets for detector (2026-06-25 재계산)

위 marker world center와 fuel_door/fuel_cap/fuel_port_hole world 좌표의 차이를 위 frame convention으로
계산했다.

- marker_to_door_xyz: [0.227924, 0.026725, -0.259677]
- marker_to_cap_xyz:  [0.230000, 0.050000, -0.240000]
- marker_to_hole_xyz: [0.223763, 0.209541, -0.216770]

주의: marker_to_hole_xyz는 hole center가 아니라 mouth surface 기준이다.
현재 multi_robot_oiling 코드가 hole 모드에서 apply_mouth_offset=True로 FUEL_PORT_DEPTH/2(0.05m) 만큼
안쪽 보정을 하기 때문에, hole_world에 PORT_OUTWARD_NORMAL_UNIT * 0.05를 더한 mouth surface 좌표로
역산해서 offset을 계산했다.

참고: door 모드도 다른 모드와 동일하게 marker 기반 lock(`door_lock_acquirer`)을 쓴다. 다만
marker_to_door_xyz는 world.usda의 닫힘(rest pose) 좌표 기준으로 계산되어 있어서, 카메라로 다시
측정해도 항상 "닫혀 있었다면 여기"라는 닫힘 등가 좌표가 나온다(문이 실제로 열려도 회전을 반영하지
않음) - lock 자체는 동작하지만 게이트 기준값도 닫힘 좌표(`task.door_world_position`)를 써야 한다.

## 검증 필요
- [LOCK REJECT] 로그의 `world_pt`가 fuel_cap_world/fuel_port_hole_world 근처(0.35m, 0.18m 게이트 이내)로
  들어오는지 실제 Isaac Sim 실행으로 확인할 것.
- 게이트를 계속 벗어나면, 그 로그의 world_pt와 기준값의 차이만큼 marker_to_*_xyz를 미세 조정.
