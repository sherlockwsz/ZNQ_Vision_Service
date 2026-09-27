# TwinCAT + 大恒相机 + Python/HALCON 视觉服务（生产版）

## 1. 这套文件做了什么

本目录把你提供的两个“文件夹离线程序”改造成单帧工业运行结构：

- PLC 仍然只决定**什么时候测量**，沿用现有 `GVL_ScrewVision.stInterface` / `GVL_CoaxVision.stInterface` 的 `RequestId -> ResultId` 握手。
- `Python Vision Service` 是唯一视觉计算入口，负责两台大恒相机、YOLO、HALCON 和 ADS；原 TCP 图像模块仅保留备用，不是本阶段检测链依赖。
- TCP 未启动、端口占用、WPF 未运行或 JPEG 编码失败，均不影响 Coax ADS 测量链。
- 螺钉算法的筛框、Spring 选择、旋转角公式保持不变，只把 `cv2.imread()` 改成内存帧。
- 同轴度算法的全视野缩小搜索、Shape Model、原图 Metrology、Fallback 和质量阈值保持不变；HALCON 返回圆心与检测状态，Python 使用 `config.toml` 的唯一标定参数源计算毫米结果。

## 2. 文件

- `vision_service.py`：主进程、ADS 轮询、双视觉 worker、事务发布。
- `camera_manager.py`：Galaxy SDK 相机唯一管理入口；软触发单帧。
- `screw_ai.py`：原螺钉算法的内存帧封装。
- `coax_halcon.py`：HALCON HDevEngine 调用与结果封装。
- `detect_coax.hdvp`：生产单帧 HALCON procedure。
- `ads_bridge.py`：对接现有 TwinCAT GVL 接口。
- `tcp_image_server.py`：向 WPF 提供最新标注 JPEG。
- `WpfVisionImageClient.cs`：可直接移植到现有 WPF 项目的只读 TCP 客户端类。
- `best.pt`：你上传的模型原文件。
- `config.toml`：ADS、相机、TCP、轴映射配置。

## 3. TwinCAT 改动

输出包里的 `TwinCAT/ZNQ_MoveCtrl_VisionCameraMode.zip` 只把：

- `GVL_ProcessConfig.stVision.eMode` 从 `Simulation` 改成 `Camera`
- `GVL_ProcessConfig.stCoaxVision.eVisionMode` 从 `Simulation` 改成 `Camera`

原有 `FB_VisionRequest`、`FB_ScrewAngleVision`、`FB_CoaxVision` 和机械流程不重写，因为它们已经具备独立 RequestId/ResultId、超时、结果有效性和双任务隔离。

## 4. 现场安装顺序

1. 安装大恒 Galaxy SDK，并确认 Galaxy Viewer 能同时枚举两台相机。
2. 安装 Galaxy SDK 附带的 Python `gxipy`。
3. 安装 HALCON 24.11 Runtime/Development，并确认对应 HALCON/Python 可以 `import halcon`，许可证正常。
4. 建议 Python 3.11；执行 `pip install -r requirements.txt`。
5. 在 TwinCAT 中打开修改后的工程，先 Build 再 Activate Configuration / Login。
6. 编辑 `config.toml`：
   - 同机运行可先留空 `ads.ams_net_id`；如果自动解析失败，填 TwinCAT Runtime 的 AMS Net ID。
   - 最终投产建议把两台相机真实 SN 填入 `serial`，不要长期依赖型号匹配。
   - 曝光和增益如果已经在 Galaxy Viewer/UserSet 中固定，可保持 `0/-1` 让服务不覆盖。
7. 运行 `run_vision_service.bat`；日志应出现两条 pipeline ready 和 ADS connected。
8. PLC 发出 `bMeasureRequest + udiRequestId` 后，服务只触发对应相机一帧，并以相同 `udiResultId` 提交结果。

## 5. 结果映射

### 螺钉

- `fDetectedAngle`：两个螺钉中心连线的原始图像角，仅诊断/显示。
- `fCorrectionAngle`：原程序 `calculate_angle_for_rotation()` 的返回值经过 `screw_mapping` 的机械轴方向/零偏映射后送 PLC；默认 `+1 / 0°`，因此默认值与原算法完全一致。
- `bDetected`：检测到 2 个 Screw 且成功选中 Spring。
- `fConfidence`：3 个被实际选中目标的最低置信度，仅诊断，不参与原算法筛选或角度判定。

### 同轴度（Coax Phase 1）

- HALCON 只返回有效圆的 `CenterColumn / CenterRow / DetectionOK`。
- Python 读取 `coax_algorithm.reference_x_px`、`reference_y_px` 和唯一的 `mm_per_pixel`，计算 `fDeltaX`、`fDeltaY` 与二维径向偏差 `fCoaxiality`。
- 坐标固定为图像坐标：`+DeltaX` 向右，`+DeltaY` 向下；本阶段不映射 AdjustmentX/Y，也不控制运动轴。
- `reference_x_px=-1.0`、`reference_y_px=-1.0` 表示暂用图像中心；`mm_per_pixel=0.0025` 也是临时调试值。投产前必须用标准件标定并替换这三个值。
- 原 `coax_mapping.delta_x_sign / delta_y_sign` 仅为后续阶段保留，本阶段不应用。

## 6. TCP 图像协议（给 WPF）

WPF 建立 TCP 长连接到 `50010`，发送一行：

- `GET screw\n`
- `GET coax\n`
- `PING\n`

响应：

1. 4 字节 big-endian JSON header 长度
2. UTF-8 JSON header
3. 4 字节 big-endian JPEG 长度
4. JPEG 数据

这里返回的是**最近一次 PLC 触发检测的标注图**，WPF 不会因此触发相机。

## 7. 关于 EXE

执行 `build_exe.bat` 可生成 `dist\VisionService\VisionService.exe`。Galaxy SDK 和 HALCON 含原生 DLL/许可证，最稳妥的现场方式是先在目标电脑安装对应官方 Runtime/SDK，再运行 EXE；不要把相机/HALCON DLL 随意从开发机拷过去。

## 8. 投产前检查

- 两台相机 SN 固定并记录。
- 两块网卡/10GigE 链路在 Galaxy Viewer 中无丢包。
- 两台相机均为 Mono8（服务会拒绝非 uint8，避免静默改变算法输入）。
- TwinCAT 两视觉模式均为 Camera。
- 逐项验证 Screw RequestId/ResultId、Coax RequestId/ResultId。
- 使用标准件标定 `reference_x_px`、`reference_y_px`、`mm_per_pixel`，并验证图像坐标下的偏差值；Adjustment 坐标映射留待后续阶段。
- 人工放置多个已知角度样件，确认 `fCorrectionAngle` 与 DamperRotation 实际正方向一致。
- 低速验证 `screw_mapping.correction_sign`；若算法定义的顺时针正方向与 DamperRotation 机械正方向相反，只改映射符号，不改角度算法。
- WPF 只读 TCP 图像，不直接连接相机。
