**발화자 인식 기반 PTZ 카메라 자동 추적 시스템**

영상 기반 객체 인식 기술과 PTZ(Pan-Tilt-Zoom) 카메라 제어를 결합하여, 발화자를 자동으로 추적하는 시스템입니다.
YOLO 기반 객체 탐지와 마이크 탐지를 활용하여 발화자를 인식하고, 카메라가 자동으로 해당 인물을 따라가도록 구현하였습니다.

현재는 다음 기능까지 개발이 완료된 상태입니다.

- YOLO 기반 사람 및 마이크 객체 탐지
- EMA(Exponential Moving Average) 기반 위치 안정화
- Sliding Window 기반 시계열 필터링 적용
- 실시간 영상 처리 및 추적 기능 구현
  
**프로젝트 목적**

기존의 고정 카메라 촬영 환경에서는 발표자 또는 발화자의 움직임에 따라 화면 구도가 자주 벗어나거나 수동 조작이 필요했습니다.
본 프로젝트는 이러한 문제를 해결하기 위해:

사람과 마이크를 실시간으로 인식하고
발화자를 자동으로 판단한 뒤
PTZ 카메라를 자동 제어하여
발화자를 지속적으로 화면 중앙에 유지하는 시스템

을 목표로 개발되었습니다.

활용 분야:

- 강의 촬영 자동화
- 회의실 발표 추적
- 방송 및 스트리밍 자동 촬영
- 스마트 강의실 시스템

**사용 기술**

**Language**
- Python
  
**AI / Vision**

- YOLOv8
- OpenCV

**Tracking & Filtering**

- EMA (Exponential Moving Average)
- Sliding Window Filtering

**Camera Control**

- PTZ Camera
- VISCA Protocol

**treaming**

- RTSP
