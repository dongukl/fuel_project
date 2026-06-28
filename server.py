#!/usr/bin/env python3
"""
AutoFuel FastAPI server.

Isaac Sim stdout 를 SSE 로 브라우저에 스트리밍하고,
POST /start 로 /start_fueling ROS2 토픽을 발행합니다.

실행 방법:
  isaac-sim.sh --headless -- cobot3_ws/isaacpjt/M0609/multi_robot_oiling.py 2>&1 \
    | uvicorn server:app --host 0.0.0.0 --port 8000

개발/테스트 (파이프 없이):
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
  (별도 터미널에서: echo "[완료] A/B 전체 시퀀스 종료" | nc localhost 8000 등)
"""

import asyncio
import sys
import threading
from collections import deque
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

# ── ROS2 (rclpy 없을 때는 graceful degradation) ────────────────────────────
_ros_available = False
_ros_node = None

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool as RosBool, Float64 as RosFloat64

    class _FuelingTrigger(Node):
        def __init__(self):
            super().__init__("autofuel_server_node")
            self._pub_start  = self.create_publisher(RosBool,     "/start_fueling", 10)
            self._pub_stop   = self.create_publisher(RosBool,     "/stop_fueling",  10)
            self._pub_target = self.create_publisher(RosFloat64,  "/fuel_target",   10)

        def set_target(self, liters: float):
            msg = RosFloat64(data=float(liters))
            self._pub_target.publish(msg)
            self.get_logger().info(f"/fuel_target → {liters:.2f} L 발행")

        def trigger(self):
            msg = RosBool(data=True)
            self._pub_start.publish(msg)
            self.get_logger().info("/start_fueling → True 발행")

        def stop(self):
            msg = RosBool(data=True)
            self._pub_stop.publish(msg)
            self.get_logger().info("/stop_fueling → True 발행")

    _ros_available = True
except ImportError:
    pass


# ── SSE 브로드캐스트 ────────────────────────────────────────────────────────
_clients: List[asyncio.Queue] = []
_event_loop: asyncio.AbstractEventLoop | None = None
_msg_history: deque = deque(maxlen=200)  # 새 브라우저 연결 시 최근 메시지 재전송용


def _broadcast(line: str) -> None:
    """stdin 읽기 스레드에서 호출 → 모든 SSE 클라이언트 큐에 라인 전달."""
    if _event_loop is None:
        return
    _msg_history.append(line)
    for q in list(_clients):
        _event_loop.call_soon_threadsafe(q.put_nowait, line)


def _stdin_reader() -> None:
    """Isaac Sim stdout (stdin 으로 파이프됨)을 라인 단위로 읽어 브로드캐스트."""
    for raw in sys.stdin:
        line = raw.rstrip()
        if line:
            print(f"[server] {line}", flush=True)   # 서버 콘솔에도 출력
            _broadcast(line)


# ── FastAPI lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    global _event_loop, _ros_node

    _event_loop = asyncio.get_running_loop()

    # stdin 읽기 스레드 시작
    reader = threading.Thread(target=_stdin_reader, daemon=True)
    reader.start()

    # ROS2 스핀 스레드 시작
    if _ros_available:
        def _ros_spin():
            global _ros_node
            rclpy.init()
            _ros_node = _FuelingTrigger()
            rclpy.spin(_ros_node)
            rclpy.shutdown()

        threading.Thread(target=_ros_spin, daemon=True).start()

    yield   # 서버 실행 중

    if _ros_available and _ros_node:
        _ros_node.destroy_node()


app = FastAPI(title="AutoFuel Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── 엔드포인트 ──────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    """receipt.html 을 직접 서빙. 브라우저에서 http://<서버IP>:8000 으로 접속."""
    return FileResponse("receipt.html", media_type="text/html")


@app.get("/events")
async def events():
    """Isaac Sim stdout 라인을 SSE(text/event-stream) 로 브라우저에 스트리밍."""
    q: asyncio.Queue[str] = asyncio.Queue()
    _clients.append(q)

    async def _stream():
        try:
            yield "data: __connected__\n\n"   # 초기 핸드셰이크
            # 브라우저가 늦게 연결된 경우를 위해 최근 메시지 이력을 재전송
            for hist_line in list(_msg_history):
                yield f"data: {hist_line}\n\n"
            # 이후 실시간 스트림
            while True:
                line = await q.get()
                yield f"data: {line}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in _clients:
                _clients.remove(q)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",          # nginx 프록시 버퍼링 방지
            "Connection": "keep-alive",
        },
    )


@app.post("/start")
async def start_fueling(request: Request):
    """목표 주유량을 받아 파일 트리거로 Isaac Sim 에 전달한다.
    uvicorn 프로세스는 ROS2 환경이 소싱되지 않을 수 있으므로
    /tmp/autofuel_start 파일을 생성하는 방식으로 신호를 전달한다.
    Isaac Sim 루프가 이 파일을 발견하면 주유 시퀀스를 시작한다."""
    import pathlib
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    target_liters = float(body.get("target_liters", 0.0))

    # 파일에 목표 주유량 기록 → Isaac Sim 루프가 읽고 삭제
    pathlib.Path("/tmp/autofuel_start").write_text(str(target_liters))

    if _ros_available and _ros_node:
        _ros_node.set_target(target_liters)
        _ros_node.trigger()
        return {"status": "ok", "ros": True, "target_liters": target_liters}
    return {"status": "ok", "ros": False, "target_liters": target_liters,
            "note": "파일 트리거(/tmp/autofuel_start) 사용"}


@app.post("/stop")
async def stop_fueling():
    """주유 정지 명령을 /stop_fueling ROS2 토픽(std_msgs/Bool true)으로 발행.
    Isaac Sim 쪽에서 이 토픽을 구독해 A 로봇의 insert 단계를 즉시 종료시킨다."""
    if _ros_available and _ros_node:
        _ros_node.stop()
        return {"status": "ok", "ros": True}
    return {"status": "ok", "ros": False, "note": "rclpy 미설치 — ROS 발행 건너뜀"}


@app.get("/health")
async def health():
    return {
        "status": "running",
        "ros_available": _ros_available,
        "sse_clients": len(_clients),
    }
